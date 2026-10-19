"""Validated CED experiment controls shared by all scheduler TP ranks."""


def validate_ced_options(options):
    if not isinstance(options, dict):
        raise TypeError("CED options must be an object")
    fields = {
        "graph_revision": ("SGLANG_DSV41_CED_TAIL_GRAPH_REVISION", int, 0, 1000),
        "batch_size": ("SGLANG_DSV41_CED_TAIL_BATCH_SIZE", int, 1, 16),
        "max_wait_ms": ("SGLANG_DSV41_CED_TAIL_MAX_WAIT_MS", int, 0, 5000),
        "encoder_graph": ("SGLANG_DSV41_CED_TAIL_ENCODER_GRAPH", bool, 0, 1),
        "encoder_exact_rows": ("SGLANG_DSV41_CED_TAIL_ENCODER_EXACT_ROWS", bool, 0, 1),
        "late_graph": ("SGLANG_DSV41_CED_TAIL_LATE_GRAPH", bool, 0, 1),
        "graph_check_every": ("SGLANG_DSV41_CED_TAIL_GRAPH_CHECK_EVERY", bool, 0, 1),
        "graph_check": ("SGLANG_DSV41_CED_TAIL_GRAPH_CHECK", bool, 0, 1),
    }
    result = {}
    for key, value in options.items():
        if key not in fields:
            raise ValueError(f"Unknown CED option: {key}")
        name, kind, low, high = fields[key]
        if type(value) is not kind or not low <= value <= high:
            raise ValueError(f"Invalid {key}: {value!r}")
        result[name] = value
    return result
