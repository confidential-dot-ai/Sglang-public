"""Failed memory registration must stop PD startup, not become a transfer timeout."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMooncakeRegistration(CustomTestCase):
    def test_nvlink_transfer_selects_rank_device_and_restores_context(self):
        import torch

        wrapper = object.__new__(MooncakeTransferEngine)
        wrapper.gpu_id = 1
        wrapper.engine = Mock()
        context = MagicMock()
        active = []
        context.__enter__.side_effect = lambda: active.append(1)
        context.__exit__.side_effect = lambda *args: active.pop() and False

        def transfer(*args):
            self.assertEqual(active, [1])
            return 0

        wrapper.engine.batch_transfer_sync_write.side_effect = transfer
        wrapper.engine.transfer_sync_write.side_effect = transfer
        with (
            patch.dict(os.environ, {"MC_INTRANODE_NVLINK": "true"}),
            patch.object(torch.cuda, "device", return_value=context) as device,
        ):
            self.assertEqual(wrapper.batch_transfer_sync("peer", [1], [2], [4]), 0)
            self.assertEqual(wrapper.transfer_sync("peer", 1, 2, 4), 0)
            self.assertEqual(device.call_count, 2)
            device.assert_called_with(1)
        self.assertEqual(active, [])

    def manager(self, status):
        manager = object.__new__(MooncakeKVManager)
        manager.engine = object.__new__(MooncakeTransferEngine)
        manager.engine.engine = Mock()
        manager.engine.engine.batch_register_memory.return_value = status
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[0x1000],
            kv_data_lens=[1024],
            aux_data_ptrs=[0x2000],
            aux_data_lens=[128],
            state_data_ptrs=[[0x1000, 0x3000]],
            state_data_lens=[[1024, 512]],
        )
        return manager

    def test_success_registers_each_distinct_region_once(self):
        manager = self.manager(0)
        manager.register_buffer_to_engine()
        manager.engine.engine.batch_register_memory.assert_called_once_with(
            [0x1000, 0x2000, 0x3000], [1024, 128, 512]
        )

    def test_kv_registration_failure_is_fatal(self):
        for status in (-1, 1):
            with self.subTest(status=status):
                manager = self.manager(status)
                with self.assertRaisesRegex(
                    RuntimeError, "Mooncake failed to register transfer memory"
                ):
                    manager.register_buffer_to_engine()

    def test_staging_registration_failure_is_fatal(self):
        manager = self.manager(-1)
        with self.assertRaisesRegex(
            RuntimeError, "Mooncake failed to register transfer memory"
        ):
            manager._register_staging_memory(0x4000, 1024)
        manager.engine.engine.batch_register_memory.assert_called_once_with(
            [0x4000], [1024]
        )

    def test_staging_registration_success(self):
        manager = self.manager(0)
        manager._register_staging_memory(0x4000, 1024)
        manager.engine.engine.batch_register_memory.assert_called_once_with(
            [0x4000], [1024]
        )

    def test_engine_exception_is_not_swallowed(self):
        manager = self.manager(0)
        manager.engine.engine.batch_register_memory.side_effect = RuntimeError(
            "registration unavailable"
        )
        with self.assertRaisesRegex(RuntimeError, "status=-1"):
            manager.register_buffer_to_engine()


if __name__ == "__main__":
    unittest.main()
