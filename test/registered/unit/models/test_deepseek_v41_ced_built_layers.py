"""A CED-split PD prefill builds only the encoder layers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CONFIG = SimpleNamespace(num_hidden_layers=40, kv_source_layer_ids=[2, 8, 20])


def _built(ced: bool, mode: str) -> int:
    with (
        envs.SGLANG_OPT_DSV41_CED_SPLIT.override(ced),
        patch.object(
            deepseek_v4,
            "get_disagg",
            lambda: SimpleNamespace(disaggregation_mode=mode),
        ),
    ):
        return deepseek_v4._num_built_layers(CONFIG)


class TestCedBuiltLayers(CustomTestCase):
    def test_ced_prefill_stops_after_the_last_kv_source_layer(self):
        self.assertEqual(_built(True, "prefill"), 21)

    def test_ced_decode_and_unsplit_workers_build_every_layer(self):
        self.assertEqual(_built(True, "decode"), 40)
        self.assertEqual(_built(False, "prefill"), 40)
        self.assertEqual(_built(False, "null"), 40)


if __name__ == "__main__":
    unittest.main()
