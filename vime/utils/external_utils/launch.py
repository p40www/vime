"""Device-specific launch policy for the vime test/example harness.

Used only by the test and example launch utilities (`command_utils.execute_train`), not by
vime core: a `Platform` describes one accelerator for the purpose of *launching a job* — how
Ray advertises its devices, the device runtime env, and how to construct the launch command.
It adapts the core ``vime.platforms`` selection to shell commands.

Imports stay stdlib-only (torch imports are lazy) so the module is unit-testable in isolation;
the actual `exec_command` calls live in `command_utils`.

NPU adaptation seam (only reached when platform.name == "npu", the CUDA path is untouched):

* ``_apply_npu_train_args`` -- a fixed baseline (the three flags every NPU test needs) plus
  per-test ``VIME_NPU_TRAIN_ARGS`` (in-place overwrite / append) and
  ``VIME_NPU_REMOVE_FLAGS`` (delete a flag together with its values).
* ``_rewrite_npu_paths`` -- rewrite hardcoded ``/root/models``, ``/root/datasets`` and
  ``*_torch_dist`` reference paths to the ``HF_HOME`` asset root, because NPU skips the
  HF -> torch_dist conversion and loads the HF directory directly.
"""

import json
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path


# -- Platform contract ----------------------------------------------------------------------


@dataclass(frozen=True)
class Platform:
    name: str
    ray_args: str  # ray-start resource flags, "{n}"-templated with the device count
    env: dict = field(default_factory=dict)  # device runtime env (into runtime_env + raylet)

    def ray_start_args(self, num_devices: int) -> str:
        return self.ray_args.format(n=num_devices)


# -- Registry -------------------------------------------------------------------------------

PLATFORMS: dict[str, Platform] = {}


def register(platform: Platform) -> None:
    PLATFORMS[platform.name] = platform


# -- Registered platforms (add a new accelerator here) ---------------------------------------


register(Platform(name="cuda", ray_args="--num-gpus {n}"))  # default; other fields unused for cuda

register(
    Platform(
        name="npu",
        # vime requests NPU bundles, not GPU (see ray/placement_group.py), so advertise
        # the custom NPU resource rather than Ray GPU capacity.
        ray_args="--num-gpus 0 --resources '{{\"NPU\": {n}}}'",
        env={
            "PYTHONPATH": (
                "/root/Megatron-LM:/root/vime:"
                "/root/MegatronAdaptor:/root/TransformerEngineNPU:"
                "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:"
                "/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge"
            ),
            "LD_LIBRARY_PATH": (
                "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:"
                "/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/ascend-toolkit/latest/lib64:"
                "/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:"
                "/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:"
                "/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/:"
                "/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:"
                "/usr/local/Ascend/cann/lib64:/usr/local/Ascend/cann/aarch64-linux/devlib"
            ),
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
            "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
            "HCCL_CONNECT_TIMEOUT": "7200",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "HYDRA_FULL_ERROR": "1",
            "TRANSFORMERS_VERBOSITY": "error",  # silence transformers image-processing log spam
        },
    )
)


# -- Resolver + launcher (platform-agnostic; unchanged when adding a platform) ---------------


def current_platform() -> Platform:
    """Adapt the selected core runtime platform to the launch contract."""
    from vime.platforms import current_platform as current_runtime_platform

    selected = current_runtime_platform().name
    try:
        return PLATFORMS[selected]
    except KeyError as exc:
        raise ValueError(f"No launcher adapter is registered for platform {selected!r}") from exc


def runtime_env_vars(platform: Platform, extra_env: dict | None = None) -> dict:
    """Device environment shared by CLI launches and externally managed Ray jobs."""
    extra_env = extra_env or {}
    diagnostics = {
        key: os.environ[key]
        for key in ("VIME_TEST_DIAGNOSTICS_DIR", "VIME_NPU_MEMORY_DIAGNOSTICS")
        if key in os.environ
    }
    all_env = {**platform.env, **diagnostics, **extra_env}
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH"):
        if key in platform.env and key in extra_env:
            paths = str(extra_env[key]).split(os.pathsep) + str(platform.env[key]).split(os.pathsep)
            all_env[key] = os.pathsep.join(dict.fromkeys(path for path in paths if path))
    all_env["VIME_PLATFORM"] = platform.name
    return all_env


# -- NPU train-args seam --------------------------------------------------------------------

_TRAIN_ARGS_ENV = "VIME_NPU_TRAIN_ARGS"
_REMOVE_FLAGS_ENV = "VIME_NPU_REMOVE_FLAGS"
_TORCH_DIST_SUFFIX = "_torch_dist"

# Baseline applied to every NPU test before per-test declarations. These are the three flags
# every NPU training job needs; per-test entries only declare what differs (memory bounds,
# removals). Declared as tokens so the JSON value keeps its own quoting.
_DEFAULT_NPU_TRAIN_ARGS: tuple[str, ...] = (
    "--use-flash-attn",
    "--no-gradient-accumulation-fusion",
    "--vllm-additional-config",
    '{"weight_nz_mode": 0}',
)


def _json_list_env(name: str) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a JSON list of flag names, got: {raw!r}") from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a JSON list of strings, got: {raw!r}")
    return value


def _split_args(text: str, *, source: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError as exc:
        if source.startswith("VIME_NPU_"):
            raise ValueError(f"{source} is not shell-parseable ({exc}): {text!r}") from exc
        print(f"[npu-args] warning: cannot shlex-parse {source} ({exc}); falling back to whitespace split")
        return text.split()


def _join_args(tokens: list[str]) -> str:
    return shlex.join(tokens)


def _flag_names(tokens: list[str]) -> list[str]:
    return [token.split("=", 1)[0] for token in tokens if token.startswith("--")]


def _flag_tokens_for(tokens: list[str], name: str) -> list[str]:
    """The flag token plus ALL its value tokens (greedy until the next --flag)."""
    out: list[str] = []
    for index, token in enumerate(tokens):
        if token.split("=", 1)[0] != name:
            continue
        out.append(token)
        if "=" not in token:
            for value in tokens[index + 1 :]:
                if value.startswith("--"):
                    break
                out.append(value)
        break
    return out


def _replace_flag(
    tokens: list[str], replacement: list[str], name: str, *, warn_if_missing: bool = False
) -> list[str]:
    """Overwrite `name` in place with `replacement` (empty list = delete).

    The flag and all of its value tokens are replaced at the original position so the order
    of unrelated flags is preserved; duplicate occurrences collapse into one. When the flag
    is absent the replacement is appended (or, for removals, a diagnostic is printed).
    """
    out: list[str] = []
    old_values: list[str] = []
    index, seen = 0, False
    while index < len(tokens):
        token = tokens[index]
        if token.split("=", 1)[0] == name:
            if "=" in token:
                if not old_values:
                    old_values = [token.split("=", 1)[1]]
                index += 1
            else:
                values: list[str] = []
                cursor = index + 1
                while cursor < len(tokens) and not tokens[cursor].startswith("--"):
                    values.append(tokens[cursor])
                    cursor += 1
                if not old_values:
                    old_values = values
                index = cursor
            if not seen:
                out.extend(replacement)
                seen = True
            continue
        out.append(token)
        index += 1

    if seen:
        if replacement:
            new_values = " ".join(replacement[1:]) if len(replacement) > 1 else ""
            print(f"[npu-args] replace: {name}: {' '.join(old_values)} -> {new_values}")
        else:
            print(f"[npu-args] remove ok: {name}")
        return out

    out.extend(replacement)
    if replacement:
        new_values = " ".join(replacement[1:]) if len(replacement) > 1 else ""
        print(f"[npu-args] append: {name} {new_values}".rstrip())
    if warn_if_missing:
        print(f"[npu-args] remove skipped (not present): {name}")
    return out


def _validate_declaration(tokens: list[str]) -> None:
    """Reject malformed VIME_NPU_TRAIN_ARGS at startup instead of after a long CI run."""
    consumed: set[int] = set()
    seen: set[str] = set()
    for index, token in enumerate(tokens):
        if not token.startswith("--"):
            continue
        name = token.split("=", 1)[0]
        if name in seen:
            raise ValueError(
                f"{_TRAIN_ARGS_ENV} declares {name!r} twice; the second occurrence would be ignored"
            )
        seen.add(name)
        if "=" in token:
            continue
        for cursor in range(index + 1, len(tokens)):
            if tokens[cursor].startswith("--"):
                break
            consumed.add(cursor)
    strays = [
        token
        for index, token in enumerate(tokens)
        if index not in consumed and not token.startswith("--")
    ]
    if strays:
        raise ValueError(
            f"{_TRAIN_ARGS_ENV} has tokens not attached to any --flag (typo?): {strays!r}"
        )


def _apply_baseline(tokens: list[str]) -> list[str]:
    baseline = list(_DEFAULT_NPU_TRAIN_ARGS)
    present = set(_flag_names(tokens))
    for name in _flag_names(baseline):
        if name in present:
            continue
        group = _flag_tokens_for(baseline, name)
        tokens = tokens + group
        print(f"[npu-args] baseline: {_join_args(group)}")
    return tokens


def _apply_npu_train_args(train_args: str) -> str:
    """Apply the NPU baseline plus per-test declarations to a train_args string."""
    tokens = _apply_baseline(_split_args(train_args, source="train_args"))
    for name in _json_list_env(_REMOVE_FLAGS_ENV):
        tokens = _replace_flag(tokens, [], name, warn_if_missing=True)

    extra = os.environ.get(_TRAIN_ARGS_ENV, "").strip()
    if extra:
        extra_tokens = _split_args(extra, source=_TRAIN_ARGS_ENV)
        _validate_declaration(extra_tokens)
        for name in _flag_names(extra_tokens):
            tokens = _replace_flag(tokens, _flag_tokens_for(extra_tokens, name), name)

    result = _join_args(tokens)
    print(f"[npu-args] final: {result}")
    return result


# -- NPU path seam --------------------------------------------------------------------------

_ROOT_PATH_MAP = (("/root/models", "models"), ("/root/datasets", "datasets"))


def _rewrite_npu_paths(text: str) -> str:
    """Rewrite hardcoded /root asset paths to the HF_HOME asset root."""
    hf_home = os.environ.get("HF_HOME", "").strip().rstrip("/")
    if not hf_home or hf_home == "/root":
        print("[npu-paths] HF_HOME not set (or still /root); skipping path rewrite")
        return text

    for old, subdir in _ROOT_PATH_MAP:
        new = f"{hf_home}/{subdir}"
        # Function replacement: a plain string replacement would interpret backslashes
        # in Windows-style paths as escape sequences.
        text, count = re.subn(rf"{re.escape(old)}(?=/|[\s'\"]|$)", lambda _match, value=new: value, text)
        if count:
            print(f"[npu-paths] {old} -> {new} ({count}x)")
    return _redirect_torch_dist_ref_load(text, Path(hf_home) / "models")


def _redirect_torch_dist_ref_load(text: str, hf_models: Path) -> str:
    """Point --ref-load at the HF directory when the converted artifact is absent."""

    def _replace(match: re.Match) -> str:
        value = match.group(1)
        path = value.rstrip("/")
        name = Path(path).name
        if not name.endswith(_TORCH_DIST_SUFFIX):
            return match.group(0)
        if Path(path).exists():
            return match.group(0)
        model_dir = hf_models / name[: -len(_TORCH_DIST_SUFFIX)]
        if model_dir.is_dir():
            print(f"[npu-paths] ref-load: {path} -> {model_dir}")
            return f"--ref-load {model_dir}"
        print(f"[npu-paths] WARN ref-load: neither {path} nor {model_dir} exists")
        return match.group(0)

    return re.sub(r"--ref-load[= ](\S+_torch_dist/?)", _replace, text)


# -- Launcher -------------------------------------------------------------------------------


def launch_commands(
    platform: Platform,
    train_args: str,
    num_devices: int,
    megatron_model_type: str | None,
    repo_base_dir: str,
    extra_env: dict | None = None,
    external_ray: bool = False,
    master_addr: str = "127.0.0.1",
    ray_running: bool = False,
) -> list:
    """Ordered shell commands to launch a training job on `platform` (pure; caller execs).

    Driven entirely by platform data (ray resources + device env), so any non-CUDA platform
    works with no change here. The caller checks Ray liveness; existing clusters
    and independently owned serving processes must survive trainer restarts.
    """
    # Test helpers prepend their import paths without hiding the platform's
    # adaptor/CANN modules or shared libraries from the raylet and its workers.
    all_env = runtime_env_vars(platform, extra_env)
    cmds: list = []
    if not external_ray and not ray_running:
        # Export the device env BEFORE `ray start` so the raylet and the actors it spawns
        # (e.g. vLLM engines) inherit it; a job's runtime_env does not reach those actors.
        exports = "".join(f"export {k}={shlex.quote(str(v))} && " for k, v in all_env.items())
        cmds.append(
            f"{exports}export PYTHONUNBUFFERED=1 && ray start --head --node-ip-address {master_addr} "
            f"{platform.ray_start_args(num_devices)} --disable-usage-stats"
        )

    runtime_env = {"env_vars": {"no_proxy": f"127.0.0.1,{master_addr}", "MASTER_ADDR": master_addr, **all_env}}
    src = f'source "{repo_base_dir}/scripts/models/{megatron_model_type}.sh" && ' if megatron_model_type else ""
    model_args = "${MODEL_ARGS[@]}" if megatron_model_type else ""
    if platform.name == "npu":
        # Per-test arg declarations first, so paths inside them are rewritten too.
        train_args = _apply_npu_train_args(train_args)
        train_args = _rewrite_npu_paths(train_args)
    cmds.append(
        f"export no_proxy=127.0.0.1 && export PYTHONUNBUFFERED=1 && {src}"
        f'ray job submit --address="http://127.0.0.1:8265" '
        f"--runtime-env-json={shlex.quote(json.dumps(runtime_env))} "
        f"-- python3 train.py {model_args} {train_args}"
    )
    return cmds
