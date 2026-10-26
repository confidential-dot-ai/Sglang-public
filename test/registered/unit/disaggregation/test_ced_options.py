import pytest

from sglang.srt.disaggregation.ced_options import validate_ced_options


@pytest.mark.parametrize(
    "options",
    [
        None,
        [],
        {"batch_size": True},
        {"batch_size": 0},
        {"batch_size": 17},
        {"max_wait_ms": -1},
        {"max_wait_ms": 5001},
        {"late_graph": 1},
        {"encoder_graph": 1},
        {"encoder_exact_rows": 1},
        {"graph_check": "true"},
        {"unknown": 1},
    ],
)
def test_invalid_options_are_rejected(options):
    with pytest.raises((ValueError, TypeError)):
        validate_ced_options(options)


def test_partial_update_preserves_unspecified_controls():
    assert validate_ced_options({"batch_size": 8, "late_graph": True}) == {
        "SGLANG_DSV41_CED_TAIL_BATCH_SIZE": 8,
        "SGLANG_DSV41_CED_TAIL_LATE_GRAPH": True,
    }


def test_http_and_ipc_preserve_structured_options_and_boolean_types():
    import msgspec
    from pydantic import TypeAdapter

    from sglang.srt.managers.io_struct import SetInternalStateReq

    options = {
        "batch_size": 8,
        "max_wait_ms": 50,
        "late_graph": True,
        "encoder_graph": True,
        "encoder_exact_rows": False,
    }
    req = TypeAdapter(SetInternalStateReq).validate_python(
        {"server_args": {"ced_tail_options": options}}
    )
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(req), type=SetInternalStateReq
    )
    assert validate_ced_options(decoded.server_args["ced_tail_options"]) == (
        validate_ced_options(options)
    )
