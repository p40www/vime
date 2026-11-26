"""Cache refresh policy shared by the weight transports."""

_PAUSE_ROLE_ORDER = {"decode": 0, "prefill": 1, "regular": 2, "encoder": 3}


def should_flush_cache(flush_cache_interval: int, weight_version: int, initial_weight_version: int = 0) -> bool:
    """Refresh at startup and every N subsequent syncs; nonpositive preserves KV.

    The first publication flushes even when resuming with a nonzero version.
    The subsequent schedule keeps its phase relative to version 1.
    A full refresh aborts requests; other updates pause them in place.
    """
    return weight_version == initial_weight_version + 1 or (
        flush_cache_interval > 0 and (weight_version - 1) % flush_cache_interval == 0
    )


def _ordered_engines(engines, *, decode_last: bool):
    """Sort engine handles by PD role: decode first for pause, decode last for resume.

    Engines that do not expose ``get_worker_type`` (older/external vLLM engines)
    keep their original order.
    """
    engines = list(engines)
    if not engines:
        return engines
    import ray

    roles = ["regular"] * len(engines)
    try:
        remote_roles = ray.get([engine.get_worker_type.remote() for engine in engines])
        roles = [str(role) for role in remote_roles]
    except Exception:
        return engines
    key = _PAUSE_ROLE_ORDER.get
    ordered = sorted(zip(engines, roles), key=lambda pair: key(pair[1], 2))
    if decode_last:
        ordered.reverse()
    return [engine for engine, _ in ordered]


def pause_engines_for_weight_update(engines, flush_cache: bool) -> None:
    """Pause generation on all engines, decode group first so prefills stop feeding it.

    Call with the same engine list that will be passed to
    :func:`resume_engines_after_weight_update`.
    """
    import ray

    mode = "abort" if flush_cache else "in_place"
    ordered = _ordered_engines(engines, decode_last=False)
    ray.get([engine.pause_generation.remote(mode=mode) for engine in ordered])


def resume_engines_after_weight_update(engines) -> None:
    """Resume generation on all engines, decode group last so KV arrives first."""
    import ray

    ordered = _ordered_engines(engines, decode_last=True)
    ray.get([engine.continue_generation.remote() for engine in ordered])
