"""DeepSeek-V4.1 Engram table formats: fp8 rows, or the mxfp4 rows of
nvidia/DeepSeek-V4.1-Flash-NVFP4 (packed e2m1, one e8m0 scale per 32 values)."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.engram import dequantize_engram_rows, engram_embed_fp4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# OCP MX e2m1 code points, codes 0-15 (sign bit 3, two exponent bits, one mantissa bit).
SPEC_E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def _config(**quant):
    return SimpleNamespace(quantization_config=dict(quant_method="fp8", **quant))


class TestEngramTableFormat(unittest.TestCase):
    def test_format_comes_from_the_checkpoint_quantization_config(self):
        layers = (1, 14)
        self.assertFalse(engram_embed_fp4(SimpleNamespace(), layers))
        self.assertFalse(engram_embed_fp4(_config(), layers))
        # The keys as nvidia/DeepSeek-V4.1-Flash-NVFP4 writes them in config.json.
        nvfp4 = _config(
            engram_embed_quant="mxfp4",
            engram_embed_block_size=32,
            engram_embed_scale_dtype="ue8m0",
            engram_embed_layers=[1, 14],
        )
        self.assertTrue(engram_embed_fp4(nvfp4, layers))
        with self.assertRaisesRegex(ValueError, "Unsupported Engram table format"):
            engram_embed_fp4(_config(engram_embed_quant="nvfp4"), layers)
        with self.assertRaisesRegex(ValueError, "Unsupported Engram table format"):
            engram_embed_fp4(
                _config(engram_embed_quant="mxfp4", engram_embed_block_size=16),
                layers,
            )
        with self.assertRaisesRegex(ValueError, "per-layer table formats"):
            engram_embed_fp4(
                _config(engram_embed_quant="mxfp4", engram_embed_layers=[1]), layers
            )

    def test_mxfp4_rows_decode_low_nibble_first_with_e8m0_scales(self):
        # Byte k holds code 2k in its low nibble and code 2k + 1 in its high one,
        # so one 32-value block walks all 16 codes twice.
        codes = torch.arange(32) % 16
        packed = (codes[0::2] | (codes[1::2] << 4)).to(torch.uint8)
        weight = packed.repeat(2).view(1, 32)  # dim 64: two scale blocks
        exps = torch.tensor([[127, 129]], dtype=torch.uint8)  # 2**0, 2**2
        scale = exps.view(torch.float8_e8m0fnu)
        got = dequantize_engram_rows(weight, scale, fp4=True)
        spec = torch.tensor(SPEC_E2M1 * 2)
        torch.testing.assert_close(got, torch.cat([spec, 4 * spec]).view(1, 64))
        # Negative zero survives: the sign is part of the code, not of the scale.
        self.assertTrue(torch.signbit(got[0, 8]))


if __name__ == "__main__":
    unittest.main()
