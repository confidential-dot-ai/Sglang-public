from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner import ced_graph_revision as revision


def test_revision_change_retains_old_graphs_and_reloads_only_named_module(monkeypatch):
    old = {"shape": object()}
    model = SimpleNamespace(graphs=old)
    module = object()
    monkeypatch.setitem(revision.sys.modules, "ced-test-module", module)
    reload = Mock(return_value="new module")
    monkeypatch.setattr(revision.importlib, "reload", reload)
    with envs.SGLANG_DSV41_CED_TAIL_GRAPH_REVISION.override(0):
        assert (
            revision.reload_if_requested("ced-test-module", 0, model, ("graphs",))
            is None
        )
        assert model.graphs is old
        reload.assert_not_called()
    with envs.SGLANG_DSV41_CED_TAIL_GRAPH_REVISION.override(1):
        assert (
            revision.reload_if_requested("ced-test-module", 0, model, ("graphs",))
            == "new module"
        )
        assert model._ced_retired_graphs == [old]
        assert not hasattr(model, "graphs")
        reload.assert_called_once_with(module)
