"""CPU-only contracts for the NPU train-args and path seams (no hardware needed).

These tests exercise the string surgery in ``vime.utils.external_utils.launch``:
the fixed NPU baseline, per-test overwrite / append / remove declarations, the
startup validation of malformed declarations, and the hardcoded-path rewrite.
Nothing here imports torch, vime.platforms, or a device.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from vime.utils.external_utils import launch as L

BASELINE = (
    "--use-flash-attn",
    "--no-gradient-accumulation-fusion",
    "--vllm-additional-config",
    '{"weight_nz_mode": 0}',
)
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Clear the declaration env and disable the baseline unless a test opts in."""
    monkeypatch.delenv(L._TRAIN_ARGS_ENV, raising=False)
    monkeypatch.delenv(L._REMOVE_FLAGS_ENV, raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(L, "_DEFAULT_NPU_TRAIN_ARGS", ())


@pytest.fixture
def with_baseline(monkeypatch):
    monkeypatch.setattr(L, "_DEFAULT_NPU_TRAIN_ARGS", BASELINE)


def apply(args: str) -> str:
    return L._apply_npu_train_args(args)


class TestBaseline:
    def test_baseline_appended(self, with_baseline):
        out = apply("--a 1")
        assert out.startswith("--a 1")
        for flag in ("--use-flash-attn", "--no-gradient-accumulation-fusion"):
            assert flag in out
        assert '{"weight_nz_mode": 0}' in out

    def test_baseline_not_duplicated(self, with_baseline):
        out = apply("--use-flash-attn --a 1")
        assert out.count("--use-flash-attn") == 1

    def test_test_values_win_over_baseline(self, with_baseline, monkeypatch):
        monkeypatch.setenv(L._TRAIN_ARGS_ENV, "--vllm-additional-config '{\"weight_nz_mode\": 1}'")
        out = apply("--a 1")
        assert out.count("--vllm-additional-config") == 1
        assert '{"weight_nz_mode": 1}' in out

    def test_no_env_keeps_input_when_baseline_disabled(self):
        assert apply("--a 1 --b 2") == "--a 1 --b 2"


class TestReplace:
    def test_override_in_place(self, monkeypatch):
        monkeypatch.setenv(L._TRAIN_ARGS_ENV, "--max-tokens-per-gpu 2048")
        out = apply("--a 1 --max-tokens-per-gpu 9216 --b 2")
        assert out == "--a 1 --max-tokens-per-gpu 2048 --b 2"

    def test_append_new_flag(self, monkeypatch):
        monkeypatch.setenv(L._TRAIN_ARGS_ENV, "--vllm-max-num-seqs 16")
        out = apply("--a 1")
        assert "--vllm-max-num-seqs 16" in out

    def test_equals_form(self, monkeypatch):
        monkeypatch.setenv(L._TRAIN_ARGS_ENV, "--max-tokens-per-gpu 2048")
        out = apply("--max-tokens-per-gpu=9999")
        assert "9999" not in out and "2048" in out

    def test_duplicate_occurrences_collapse(self, monkeypatch):
        monkeypatch.setenv(L._TRAIN_ARGS_ENV, "--max-tokens-per-gpu 2048")
        out = apply("--max-tokens-per-gpu 8192 --a 1 --max-tokens-per-gpu 4096")
        assert out.count("--max-tokens-per-gpu") == 1
        assert "2048" in out and "8192" not in out and "4096" not in out

    def test_untouched_segment_unchanged(self, monkeypatch):
        monkeypatch.setenv(L._TRAIN_ARGS_ENV, "--vllm-gpu-memory-utilization 0.5")
        out = apply("--a 1 --vllm-gpu-memory-utilization 0.65 --b 2 --c 3")
        assert "--a 1" in out and "--b 2" in out and "--c 3" in out

    def test_idempotent(self, monkeypatch):
        monkeypatch.setenv(L._TRAIN_ARGS_ENV, "--vllm-gpu-memory-utilization 0.5")
        once = apply("--a 1 --vllm-gpu-memory-utilization 0.65 --b 2")
        assert apply(once) == once

    def test_json_value_survives(self, monkeypatch):
        monkeypatch.setenv(
            L._TRAIN_ARGS_ENV,
            "--use-flash-attn --vllm-additional-config '{\"weight_nz_mode\": 0}' --max-tokens-per-gpu 2048",
        )
        out = apply("--a 1 --max-tokens-per-gpu 9216 --b 2")
        assert '{"weight_nz_mode": 0}' in out
        assert "2048" in out and "9216" not in out


class TestRemove:
    def test_remove_flag_with_value(self, monkeypatch):
        monkeypatch.setenv(L._REMOVE_FLAGS_ENV, '["--rollout-data-transport"]')
        out = apply("--rollout-data-transport nixl --a 1")
        assert "nixl" not in out and "--rollout-data-transport" not in out

    def test_remove_absent_is_not_fatal(self, monkeypatch):
        monkeypatch.setenv(L._REMOVE_FLAGS_ENV, '["--not-present"]')
        assert apply("--a 1 --b 2") == "--a 1 --b 2"

    def test_remove_multi_value_flag(self, monkeypatch):
        monkeypatch.setenv(L._REMOVE_FLAGS_ENV, '["--vllm-cudagraph-capture-sizes"]')
        assert apply("--a 1 --vllm-cudagraph-capture-sizes 5 10 20 40 80 --b 2") == "--a 1 --b 2"

    def test_replace_multi_value_flag(self, monkeypatch):
        monkeypatch.setenv(L._TRAIN_ARGS_ENV, "--vllm-cudagraph-capture-sizes 8")
        out = apply("--vllm-cudagraph-capture-sizes 5 10 20 40 80 --b 2")
        assert out == "--vllm-cudagraph-capture-sizes 8 --b 2"

    def test_remove_then_redeclare(self, monkeypatch):
        monkeypatch.setenv(L._TRAIN_ARGS_ENV, "--rollout-data-transport object-store")
        monkeypatch.setenv(L._REMOVE_FLAGS_ENV, '["--rollout-data-transport"]')
        out = apply("--rollout-data-transport nixl --a 1")
        assert "nixl" not in out and "object-store" in out


class TestFailLoud:
    def test_duplicate_declaration_raises(self, monkeypatch):
        monkeypatch.setenv(L._TRAIN_ARGS_ENV, "--a 1 --a 2")
        with pytest.raises(ValueError, match="twice"):
            apply("--b 1")

    def test_stray_token_raises(self, monkeypatch):
        monkeypatch.setenv(L._TRAIN_ARGS_ENV, "stray --max-tokens-per-gpu 2048")
        with pytest.raises(ValueError, match="not attached"):
            apply("--a 1")

    def test_single_dash_typo_raises(self, monkeypatch):
        monkeypatch.setenv(L._TRAIN_ARGS_ENV, "-use-flash-attn --a 1")
        with pytest.raises(ValueError, match="not attached"):
            apply("--a 1")

    def test_bad_json_raises(self, monkeypatch):
        monkeypatch.setenv(L._REMOVE_FLAGS_ENV, "not-json")
        with pytest.raises(ValueError):
            apply("--a 1")

    def test_unclosed_quote_in_env_raises(self, monkeypatch):
        monkeypatch.setenv(L._TRAIN_ARGS_ENV, "--flag 'unclosed")
        with pytest.raises(ValueError):
            apply("--a 1")

    def test_malformed_train_args_falls_back(self):
        out = apply("--a 1 'unclosed")
        assert "--a" in out and "1" in out


class TestPaths:
    def test_root_paths_rewritten(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        out = L._rewrite_npu_paths(
            "--local-dir /root/models/Qwen --prompt-data /root/datasets/gsm8k/train.parquet"
        )
        assert f"{tmp_path}/models/Qwen" in out
        assert f"{tmp_path}/datasets/gsm8k/train.parquet" in out
        assert "/root/models" not in out and "/root/datasets" not in out

    def test_path_boundary_not_rewritten(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        assert "/root/modelsX" in L._rewrite_npu_paths("--a /root/modelsX")

    def test_hf_home_unset_skips(self):
        args = "--local-dir /root/models/Qwen"
        assert L._rewrite_npu_paths(args) == args

    def test_torch_dist_redirected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        model_dir = tmp_path / "models" / "Qwen"
        model_dir.mkdir(parents=True)
        out = L._rewrite_npu_paths("--ref-load /dev/shm/Qwen_torch_dist --a 1")
        assert f"--ref-load {model_dir}" in out

    def test_torch_dist_kept_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        converted = tmp_path / "Qwen_torch_dist"
        converted.mkdir()
        args = f"--ref-load {converted} --a 1"
        assert L._rewrite_npu_paths(args) == args

    def test_torch_dist_missing_target_kept(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        args = "--ref-load /nonexistent-dir-xyz/Qwen_torch_dist"
        assert L._rewrite_npu_paths(args) == args


class TestRegistry:
    """Validate every declaration in the NPU suite registry without a device."""

    @staticmethod
    def _suites():
        path = REPO_ROOT / ".buildkite" / "npu_suites.py"
        spec = importlib.util.spec_from_file_location("npu_suites_contract", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SUITES

    def test_registry_is_not_empty(self):
        suites = self._suites()
        assert sum(len(entries) for entries in suites.values()) > 0

    def test_all_declarations_validate(self, monkeypatch):
        checked = 0
        for entries in self._suites().values():
            for entry in entries:
                env = entry[3] if len(entry) >= 4 else {}
                raw_args = env.get(L._TRAIN_ARGS_ENV, "")
                raw_remove = env.get(L._REMOVE_FLAGS_ENV, "")
                if not raw_args and not raw_remove:
                    continue
                checked += 1
                monkeypatch.setenv(L._TRAIN_ARGS_ENV, raw_args)
                monkeypatch.setenv(L._REMOVE_FLAGS_ENV, raw_remove or "")
                apply("--dry-run 1")
                if raw_remove:
                    assert isinstance(json.loads(raw_remove), list)
        assert checked >= 0
