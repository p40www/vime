"""Qwen3-0.6B PD disaggregation verification smoke for Ascend NPU.

Base branch: pr-476. The 476 line has no Mooncake dispatch yet: until the
platform wiring (VLLMLaunchPlatformOps.pd_config / NpuVLLMLaunchPlatformOps
.pd_config plus the vllm_engine call site) is added, --prefill-num-servers
keeps the hardcoded NixlConnector and fails on NPU with "NIXL is not
available".

Paths and preparation follow test_delta_disk_weight_update.py (HF_HOME-based
asset locations); the ref checkpoint follows whichever form exists
(converted torch_dist or native HF weights); the PD topology follows
test_qwen3.6_35B_A3B_pd_mooncake.py (--prefill-num-servers 1).

Runs one RL training configuration twice from identical initial weights:

- baseline: regular single-group rollout engines (PD off);
- pd:       one prefill server plus decode engines (PD on).

Verified:
1. both loops complete their rollout steps under --ci-test;
2. synchronized weights are equal on BOTH the prefill and the decode engine
   groups after update_weights (--check-weight-update-equal);
3. PD-on responses match the engine's own noise floor: on deterministic
   backends this means token-identical output, on NPU (no batch-invariant
   kernels) the cross-topology agreement must not fall below the divergence
   the engine produces between greedy copies inside one run. Set
   VIME_TEST_PD_CONSISTENCY=0 to skip the baseline comparison while debugging.
"""

import os
import shlex
import tempfile
from pathlib import Path

import vime.utils.external_utils.command_utils as U
from vime.observability.rollout_data_utils import load_debug_rollout_data

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 4
NUM_ROLLOUT_PD = 2
NUM_ROLLOUT_BASELINE = 1
TEST_ROOT = os.environ.get("HF_HOME") or "/root"
MODEL_DIR = f"{TEST_ROOT}/models/{MODEL_NAME}"
DATASET_DIR = f"{TEST_ROOT}/datasets/gsm8k"
TORCH_DIST_CKPT = f"{TEST_ROOT}/tmp/{MODEL_NAME}_torch_dist"
EXPECTED_SAMPLES = 16  # rollout_batch_size * n_samples_per_prompt


def prepare():
    U.exec_command(
        f"mkdir -p {shlex.quote(MODEL_DIR)} {shlex.quote(DATASET_DIR)} {shlex.quote(f'{TEST_ROOT}/tmp')}"
    )
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {shlex.quote(MODEL_DIR)}")
    U.exec_command("hf download --repo-type dataset zhuzilin/gsm8k " f"--local-dir {shlex.quote(DATASET_DIR)}")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst=f"{TEST_ROOT}/tmp",
        hf_checkpoint=MODEL_DIR,
    )
    # Newer vime skips torch_dist conversion on NPU (native HF load); the
    # training args then fall back to MODEL_DIR as the reference checkpoint.
    if not Path(TORCH_DIST_CKPT).is_dir():
        print(f"no torch_dist checkpoint; the run will use HF weights from {MODEL_DIR}", flush=True)


def _train_args(*, pd_enabled: bool, num_rollout: int, debug_data: Path) -> str:
    # Older vime on NPU trains from a converted torch_dist checkpoint; newer
    # vime loads HF weights natively and skips the conversion.
    ref = TORCH_DIST_CKPT if Path(TORCH_DIST_CKPT).is_dir() else MODEL_DIR
    ckpt_args = f"--hf-checkpoint {shlex.quote(MODEL_DIR)} --ref-load {shlex.quote(ref)} "

    rollout_args = (
        f"--prompt-data {shlex.quote(f'{DATASET_DIR}/train.parquet')} "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rm-type math "
        f"--num-rollout {num_rollout} "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1.0 "
        "--rollout-top-k 1 "
        "--global-batch-size 16 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.01 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    vllm_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--rollout-num-gpus 3 "
        "--vllm-gpu-memory-utilization 0.7 "
        "--vllm-max-cudagraph-capture-size 32 "
    )
    pd_args = "--prefill-num-servers 1 " if pd_enabled else ""

    verify_args = (
        "--ci-test "
        # Greedy (top-k 1) decoding shifts the vLLM-side logprobs reported to
        # training, so the default 0.1 alignment threshold does not apply to
        # this consistency-oriented smoke.
        "--ci-train-rollout-logprob-abs-diff-threshold 1.0 "
        "--check-weight-update-equal "
        f"--save-debug-rollout-data {shlex.quote(str(debug_data / 'rollout_{rollout_id}.pt'))} "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--vllm-additional-config '{\"weight_nz_mode\":0}' "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 1 "
    )

    return (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{perf_args} "
        f"{grpo_args} "
        f"{optimizer_args} "
        f"{vllm_args} "
        f"{pd_args} "
        f"{verify_args} "
        f"{misc_args} "
    )


def _run(*, pd_enabled: bool, num_rollout: int, debug_data: Path):
    U.execute_train(
        train_args=_train_args(pd_enabled=pd_enabled, num_rollout=num_rollout, debug_data=debug_data),
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


def _sample_tokens(sample):
    return [int(token) for token in sample.tokens]


def _group_key(sample):
    tokens = _sample_tokens(sample)
    response_length = int(sample.response_length)
    # The prompt token prefix (tokens minus the response tail) groups the
    # sibling samples of one prompt regardless of completion order.
    return tuple(tokens[: len(tokens) - response_length]) if response_length else tuple(tokens)


def _group_by_prompt(samples):
    grouped = {}
    for sample in samples:
        grouped.setdefault(_group_key(sample), []).append(_sample_tokens(sample))
    return grouped


def _compare_rollout_data(baseline_dir: Path, pd_dir: Path):
    baseline = load_debug_rollout_data(str(baseline_dir / "rollout_{rollout_id}.pt"), rollout_id=0)
    pd = load_debug_rollout_data(str(pd_dir / "rollout_{rollout_id}.pt"), rollout_id=0)
    assert len(baseline) == EXPECTED_SAMPLES, f"baseline saved {len(baseline)} samples, expected {EXPECTED_SAMPLES}"
    assert len(pd) == EXPECTED_SAMPLES, f"pd saved {len(pd)} samples, expected {EXPECTED_SAMPLES}"

    baseline_group = _group_by_prompt(baseline)
    pd_group = _group_by_prompt(pd)
    assert set(baseline_group) == set(pd_group), "the two runs used different prompts; check the data source order"

    def _first_divergence(tokens_a, tokens_b):
        index = 0
        while index < min(len(tokens_a), len(tokens_b)) and tokens_a[index] == tokens_b[index]:
            index += 1
        return index, min(len(tokens_a), len(tokens_b))

    # NPU engines are not request-batch invariant: greedy copies of one prompt
    # already diverge inside a single run, so exact cross-topology token
    # equality is unattainable there. Use the run's own divergence depth as the
    # noise floor and require the PD topology to stay within it. On
    # deterministic backends the floor is exact match, so the check stays
    # strict there.
    inner = []
    for prompt_tokens, responses in baseline_group.items():
        if len(responses) >= 2:
            depth, length = _first_divergence(responses[0], responses[1])
            inner.append(depth / max(length, 1))
    noise_floor = sum(inner) / len(inner) if inner else 1.0

    cross = []
    token_identical = 0
    worst = []
    for prompt_tokens in baseline_group:
        depth, length = _first_divergence(baseline_group[prompt_tokens][0], pd_group[prompt_tokens][0])
        cross.append(depth / max(length, 1))
        if depth >= length > 0:
            token_identical += 1
        else:
            worst.append((depth / max(length, 1), depth, length))

    summary = {
        "prompts": len(baseline_group),
        "samples": len(baseline),
        "token_identical_prompts": token_identical,
        "engine_noise_floor": round(noise_floor, 3),
        "pd_cross_consistency": round(sum(cross) / len(cross), 3),
    }
    print(f"PD consistency comparison: {summary}", flush=True)
    for ratio, depth, length in sorted(worst)[:2]:
        print(f"shallowest agreement: {ratio:.3f} (diverged at token {depth}/{length})", flush=True)
    assert sum(cross) / len(cross) + 1e-9 >= noise_floor, (
        f"PD consistency {sum(cross) / len(cross):.3f} is below the engine's own noise floor "
        f"{noise_floor:.3f}; this indicates a real PD transfer defect, not batching nondeterminism"
    )


def execute():
    scratch_root = Path(TEST_ROOT) / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vime_pd06b_", dir=scratch_root) as scratch:
        _run(pd_enabled=True, num_rollout=NUM_ROLLOUT_PD, debug_data=Path(scratch) / "pd")
        if not U.get_bool_env_var("VIME_TEST_PD_CONSISTENCY", "1"):
            print("VIME_TEST_PD_CONSISTENCY=0: skipping baseline comparison", flush=True)
            return
        _run(pd_enabled=False, num_rollout=NUM_ROLLOUT_BASELINE, debug_data=Path(scratch) / "baseline")
        _compare_rollout_data(Path(scratch) / "baseline", Path(scratch) / "pd")


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
