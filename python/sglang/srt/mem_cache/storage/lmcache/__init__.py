"""LMCache-backed KV cache integrations."""


def validate_connector_options(mode: str, host_memory_backend: str) -> None:
    """Validate the connector/allocator pair before exporting any KV."""
    if mode not in ("auto", "ip", "mp"):
        raise ValueError(f"Unknown LMCache mode: {mode!r}")
    if host_memory_backend not in ("default", "hardwire"):
        raise ValueError(
            f"Unknown LMCache host memory backend: {host_memory_backend!r}"
        )
    if mode == "mp" and host_memory_backend == "hardwire":
        raise ValueError(
            "Hardwire LMCache host memory requires the in-process connector; "
            "the multiprocess connector exports GPU tensors through CUDA IPC."
        )


def resolve_connector_mode(mode: str, host_memory_backend: str) -> str:
    validate_connector_options(mode, host_memory_backend)
    if mode == "auto":
        return "ip" if host_memory_backend == "hardwire" else "mp"
    return mode
