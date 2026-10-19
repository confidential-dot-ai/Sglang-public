"""Distributed group creation must consume the resolved runtime settings."""

import unittest
from unittest.mock import patch

from sglang.srt.distributed import bootstrap
from sglang.srt.runtime_context import (
    get_context,
    get_disagg,
    get_exec,
    get_parallel,
    get_server_args,
    override_platform,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBootstrapRuntimeConfig(CustomTestCase):
    def test_group_creation_uses_resolved_platform_overlap_and_pdmux(self):
        for hip in (False, True):
            for pdmux in (False, True):
                with (
                    self.subTest(hip=hip, pdmux=pdmux),
                    get_context().override_server_args(tp_size=4),
                    get_parallel().override(
                        tp_rank=0, pp_rank=0, enable_prefill_cp=True
                    ),
                    get_exec().overlap.override(enable_two_batch_overlap=True),
                    get_disagg().override(enable_pdmux=pdmux),
                    override_platform(is_hip=hip),
                    patch.object(bootstrap, "init_distributed_environment"),
                    patch.object(bootstrap, "initialize_model_parallel") as initialize,
                    patch.object(
                        bootstrap, "_tag_groups_for_flashinfer_allreduce_only"
                    ),
                    patch.object(bootstrap, "initialize_dp_attention"),
                    patch.object(bootstrap, "is_npu", return_value=False),
                ):
                    bootstrap._init_parallel_groups(
                        backend="gloo",
                        dist_init_method="unused",
                        server_args=get_server_args(),
                        gpu_id=0,
                    )
                    config = initialize.call_args.kwargs
                    self.assertEqual(config["duplicate_attn_cp_group"], hip)
                    self.assertEqual(config["duplicate_tp_group"], pdmux)


if __name__ == "__main__":
    unittest.main()
