from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.model_executor.runner import ced_graph_cache as cache_module


def test_revision_retirement_releases_only_retired_owners(monkeypatch):
    old = {"old": object()}
    active = {"active": object()}
    model = SimpleNamespace(_ced_retired_graphs=[old], _ced_encoder_graphs=active)
    sync = Mock(side_effect=lambda: None if model._ced_retired_graphs else 1 / 0)
    monkeypatch.setattr(cache_module.torch.cuda, "synchronize", sync)
    monkeypatch.setattr(cache_module.gc, "collect", Mock())
    cache_module.release_retired(model)
    assert model._ced_retired_graphs == []
    assert model._ced_encoder_graphs is active
    sync.assert_called_once()


def test_capture_reserves_memory_after_releasing_unused_blocks(monkeypatch):
    monkeypatch.setattr(cache_module, "_released_load_buffers", False)
    empty = Mock()
    monkeypatch.setattr(cache_module.torch.cuda, "empty_cache", empty)
    monkeypatch.setattr(
        cache_module.torch.cuda, "mem_get_info", lambda: (19 * 1024**3, 280 * 1024**3)
    )
    assert not cache_module.capture_has_room()
    empty.assert_called_once()
    monkeypatch.setattr(
        cache_module.torch.cuda, "mem_get_info", lambda: (21 * 1024**3, 280 * 1024**3)
    )
    assert cache_module.capture_has_room()
    empty.assert_called_once()


def test_capture_floor_follows_the_free_memory_setting(monkeypatch):
    from sglang.srt.environ import envs

    monkeypatch.setattr(cache_module, "_released_load_buffers", True)
    monkeypatch.setattr(
        cache_module.torch.cuda, "mem_get_info", lambda: (16 * 1024**3, 180 * 1024**3)
    )
    # 16 GiB free: below the default 20 GiB floor, above an 8 GiB one.
    assert not cache_module.capture_has_room()
    with envs.SGLANG_DSV41_CED_GRAPH_MIN_FREE_GB.override(8.0):
        assert cache_module.capture_has_room()
