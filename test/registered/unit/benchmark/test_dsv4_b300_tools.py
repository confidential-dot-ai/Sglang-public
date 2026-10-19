import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[4]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREFLIGHT = load_module(
    "dsv4_b300_preflight_test", "benchmark/deepseek_v4_b300/preflight.py"
)
BUNDLE = load_module(
    "dsv4_b300_bundle_test", "benchmark/deepseek_v4_b300/bundle_candidate.py"
)
TUNER = load_module("dsv4_b300_tune_test", "benchmark/deepseek_v4_b300/tune.py")
PROBE = load_module("dsv4_correctness_probe_test", "scripts/dsv4_correctness_probe.py")
WORKLOAD = load_module(
    "dsv4_workload_manifest_test",
    "benchmark/deepseek_v4_b300/workload_manifest.py",
)
COMPARE = load_module(
    "dsv4_compare_probes_test",
    "benchmark/deepseek_v4_b300/compare_probes.py",
)
MATRIX = load_module(
    "dsv4_experiment_matrix_test",
    "benchmark/deepseek_v4_b300/experiment_matrix.py",
)


def replay_payload(*, tpm=100.0, drain_tpm=98.0, ttft=1.0, itl=0.01):
    return {
        "timing": {"fixed_window_complete": True},
        "metrics": {
            "fixed_window": {
                "failed_requests": 0,
                "served_tpm_per_gpu": tpm,
            },
            "drain_inclusive": {
                "failed_requests": 0,
                "served_tpm_per_gpu": drain_tpm,
            },
        },
        "latency": {
            "fixed_window": {"ttft_p50": ttft, "token_itl_ms_p50": itl}
        },
        "scorecard": {"pass": True, "failures": []},
    }


def bundle(*, tpm=100.0, drain_tpm=98.0, ttft=1.0, itl=0.01, config=None):
    return {
        "replay": replay_payload(
            tpm=tpm, drain_tpm=drain_tpm, ttft=ttft, itl=itl
        ),
        "correctness": [{"passed": True}],
        "configuration": config or {},
    }


class Dsv4PreflightTest(unittest.TestCase):
    def test_b300_name_is_specific(self):
        self.assertTrue(PREFLIGHT.is_b300_name("NVIDIA B300"))
        self.assertTrue(PREFLIGHT.is_b300_name("NVIDIA GB300 NVL72"))
        self.assertFalse(PREFLIGHT.is_b300_name("NVIDIA B200"))

    def test_cc_gate_requires_all_three_states(self):
        state = {
            "feature": {"returncode": 0, "output": "CC feature status: ON"},
            "devtools": {"returncode": 0, "output": "DevTools Mode: OFF"},
            "multi_gpu": {"returncode": 0, "output": "Multi-GPU status: enabled"},
        }
        self.assertTrue(PREFLIGHT.cc_is_qualified(state))
        state["devtools"]["output"] = "DevTools Mode: ON"
        self.assertFalse(PREFLIGHT.cc_is_qualified(state))

    def test_inspect_records_exact_node_and_selected_island(self):
        inventory = "\n".join(
            f"{i}, NVIDIA B300, GPU-{i}, 0000:{i:02x}:00.0, 286720, 0, 600.1"
            for i in range(8)
        )

        def run(command):
            joined = " ".join(command)
            if "--query-gpu=" in joined:
                return 0, inventory
            if "--query-compute-apps=" in joined:
                return 0, ""
            if command[-1] == "-f":
                return 0, "CC feature status: ON"
            if command[-1] == "-d":
                return 0, "DevTools Mode: OFF"
            if command[-1] == "-mgm":
                return 0, "Multi-GPU status: enabled"
            if command[:3] == ["nvidia-smi", "topo", "-m"]:
                return 0, "GPU0 GPU1 NV18"
            if command[-2:] == ["rev-parse", "HEAD"]:
                return 0, "deadbeef\n"
            if command[-2:] == ["status", "--porcelain=v1"]:
                return 0, ""
            if "submodule" in command:
                return 0, ""
            if command[-1] == "--version" and command[0] == "nvcc":
                return 0, "Cuda compilation tools, release 13.0"
            if "-c" in command:
                return 0, json.dumps(
                    {
                        "python": "3.12",
                        "torch": "2.11",
                        "cuda": "13.0",
                        "nccl": [2, 28, 0],
                        "flashinfer": "0.6.14",
                        "sglang_path": str(ROOT / "python/sglang/__init__.py"),
                    }
                )
            raise AssertionError(command)

        payload = PREFLIGHT.inspect(
            repo_root=ROOT,
            selected_gpu_indices=[0, 1, 2, 3],
            expected_gpu_count=8,
            require_cc=True,
            require_idle=True,
            allow_non_b300=False,
            attestation_evidence=None,
            require_attestation=False,
            run=run,
        )
        self.assertEqual(len(payload["hardware"]["inventory"]), 8)
        self.assertEqual(payload["hardware"]["selected_gpu_indices"], [0, 1, 2, 3])
        self.assertTrue(payload["confidential_compute"]["qualified"])


class Dsv4BundleAndSelectionTest(unittest.TestCase):
    def test_bundle_validation_fails_closed(self):
        candidate = bundle()
        self.assertEqual(BUNDLE.validate_bundle(candidate), [])
        candidate["correctness"][0]["passed"] = False
        self.assertIn("correctness probe 0 did not pass", BUNDLE.validate_bundle(candidate))

    def test_variant_requires_clear_safe_end_to_end_win(self):
        bundles = {
            "control": bundle(),
            "winner": bundle(tpm=105.0, drain_tpm=100.0, ttft=1.02, itl=0.0102),
        }
        selected = TUNER.select_variant(bundles, baseline_name="control")
        self.assertEqual(selected["selected"], "winner")

    def test_variant_rejects_latency_regression(self):
        bundles = {
            "control": bundle(),
            "fast-but-slow-ttft": bundle(tpm=120.0, drain_tpm=110.0, ttft=1.2),
        }
        selected = TUNER.select_variant(bundles, baseline_name="control")
        self.assertEqual(selected["selected"], "control")

    def test_profile_rejects_non_allowlisted_environment(self):
        with self.assertRaisesRegex(ValueError, "non-allowlisted"):
            TUNER.render_profile_env({"LD_PRELOAD": "/tmp/not-allowed.so"})

    def test_collective_selector_keeps_baseline_without_two_percent_win(self):
        def payload(scale):
            return {
                "hardware_cc_state": {
                    "query_ok": True,
                    "enabled": True,
                    "devtools_mode": 0,
                },
                "correctness_max_abs_error": 0.0,
                "results": [
                    {
                        "message_mib": 1.0,
                        "pynccl": [
                            {
                                "eager_max_abs_error": 0.0,
                                "graph_max_abs_error": 0.0,
                                "graph_replay_ms": {"max_rank_ms": scale},
                            }
                        ],
                    }
                ],
            }

        selected = TUNER.select_collective(
            {"ring-ll": payload(1.0), "candidate": payload(0.99)},
            baseline_name="ring-ll",
            weights={"1MiB": 1.0},
            require_cc=True,
        )
        self.assertEqual(selected["selected"], "ring-ll")

    def test_dry_run_emits_fail_closed_profile(self):
        with TemporaryDirectory() as output_dir:
            argv = [
                "tune.py",
                "--dry-run",
                "--quick",
                "--allow-non-b300",
                "--no-require-cc",
                "--no-require-attestation",
                "--output-dir",
                output_dir,
            ]
            old_argv = sys.argv
            try:
                sys.argv = argv
                args = TUNER.parse_args()
                self.assertEqual(TUNER.Tuner(args).run(), 0)
            finally:
                sys.argv = old_argv
            profile = (Path(output_dir) / "profile.env").read_text()
            self.assertIn("export DSV4_COMPRESS_STATE_DTYPE=float32", profile)
            self.assertIn("export DSV4_ENABLE_DP_ATTENTION=0", profile)
            self.assertIn("export DSV4_ENABLE_FP4_INDEXER=0", profile)
            self.assertIn("export DSV4_FUSE_C4_FP4_STORE=1", profile)
            self.assertIn("export DSV4_ENABLE_TBO_CUDA_GRAPH=0", profile)
            report = json.loads((Path(output_dir) / "profile.json").read_text())
            self.assertEqual(report["suites"]["collectives"]["status"], "planned")


class Dsv4ExperimentMatrixTest(unittest.TestCase):
    def test_candidate_graph_is_one_variable_at_a_time(self):
        rows = MATRIX.candidate_definitions()
        MATRIX.validate_definitions(rows)
        by_name = {row["name"]: row for row in rows}
        self.assertEqual(
            by_name["graph-bs-384"]["changed_keys"], ["CUDA_GRAPH_MAX_BS"]
        )
        self.assertEqual(
            by_name["fp32-fp4-indexer-unfused"]["changed_keys"],
            ["DSV4_FUSE_C4_FP4_STORE"],
        )
        self.assertEqual(
            by_name["target-dp-attention-4"]["changed_keys"],
            ["DSV4_ENABLE_DP_ATTENTION"],
        )
        self.assertEqual(
            by_name["dspark-dp-attention-4"]["parent"],
            "target-dp-attention-4",
        )
        self.assertIn("blocked_until", by_name["tbo-on"])
        self.assertEqual(
            by_name["tbo-on"]["profile"]["DSV4_ENABLE_TBO_CUDA_GRAPH"], "1"
        )

    def test_matrix_binds_inputs_and_expands_repetitions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.json"
            manifest = root / "manifest.json"
            attestation = root / "attestation.json"
            dataset.write_text('{"requests": []}\n')
            manifest.write_text('{"contract": {}}\n')
            attestation.write_text('{"verified": true}\n')
            payload = MATRIX.build_matrix(
                repo_root=ROOT,
                run_root=root / "runs",
                dataset=dataset,
                workload_manifest=manifest,
                attestation=attestation,
                max_total_tokens=1234,
                refresh_prob=0.125,
                repetitions=2,
                include_tbo=False,
            )
            self.assertEqual(
                payload["immutable_inputs"]["dataset"]["sha256"],
                MATRIX.sha256_file(dataset),
            )
            self.assertTrue(all(len(row["runs"]) == 2 for row in payload["candidates"]))
            baseline = next(
                row for row in payload["candidates"] if row["name"] == "fp32-default"
            )
            self.assertEqual(
                baseline["runs"][0]["launch"]["environment"]["MAX_TOTAL_TOKENS"],
                "1234",
            )


class Dsv4LaunchContractTest(unittest.TestCase):
    def test_dp_attention_switch_is_complete_and_manifested(self):
        launcher = (
            ROOT / "benchmark/deepseek_v4_b300/launch_dsv4_flash_0731_tp4.sh"
        ).read_text()
        for required in (
            "--dp 4",
            "--enable-dp-attention",
            "--enable-dp-attention-local-control-broadcast",
            "--enable-dp-lm-head",
            "--moe-a2a-backend none",
            'MANIFEST_ENABLE_DP_ATTENTION="${DSV4_ENABLE_DP_ATTENTION}"',
            '"dp_attention_size": 4 if dp_attention else 1',
        ):
            self.assertIn(required, launcher)

    def test_tbo_cuda_graph_switch_is_exported_and_manifested(self):
        launcher = (
            ROOT / "benchmark/deepseek_v4_b300/launch_dsv4_flash_0731_tp4.sh"
        ).read_text()
        for required in (
            'DSV4_ENABLE_TBO_CUDA_GRAPH=${DSV4_ENABLE_TBO_CUDA_GRAPH:-${DSV4_ENABLE_TBO}}',
            'export SGLANG_ENABLE_DEEPSEEK_V4_TBO_CUDA_GRAPH="${DSV4_ENABLE_TBO_CUDA_GRAPH}"',
            'MANIFEST_ENABLE_TBO_CUDA_GRAPH="${DSV4_ENABLE_TBO_CUDA_GRAPH}"',
            '"enable_deepseek_v4_tbo_cuda_graph": (',
        ):
            self.assertIn(required, launcher)


class Dsv4ProbeTest(unittest.TestCase):
    def test_parse_completion_detects_token_cap(self):
        parsed = PROBE.parse_completion(
            {
                "choices": [
                    {"finish_reason": "length", "message": {"content": "x"}}
                ],
                "usage": {"completion_tokens": 8},
            },
            max_tokens=8,
        )
        self.assertTrue(parsed["hit_max_tokens"])

    def test_tool_call_validation(self):
        error = PROBE.validate_tool_call(
            {
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"Beijing"}',
                        }
                    }
                ]
            }
        )
        self.assertIsNone(error)

    def test_probe_comparison_requires_exact_deterministic_match(self):
        reference = {
            "passed": True,
            "cases": [
                {
                    "prompt": "p",
                    "finish_reason": "stop",
                    "content": "answer",
                    "tool_calls": [],
                }
            ],
        }
        candidate = json.loads(json.dumps(reference))
        self.assertEqual(COMPARE.compare(reference, candidate), [])
        candidate["cases"][0]["content"] = "different"
        self.assertIn(
            "deterministic output mismatch for prompt 'p'",
            COMPARE.compare(reference, candidate),
        )


class Dsv4WorkloadManifestTest(unittest.TestCase):
    def test_verify_locks_hash_and_contract(self):
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.json"
            dataset.write_text(json.dumps({"conversations": [[], []]}))
            contract = {
                "n_convs": 2,
                "concurrency": 2,
                "max_turns": 25,
                "warmup_seconds": 60,
                "measurement_seconds": 600,
                "physical_gpus": 8,
                "cache_hit_min": 0.955,
                "cache_hit_max": 0.965,
                "output_share_min": 0.007,
                "output_share_max": 0.008,
                "refresh_prob": 0.1,
            }
            manifest = {
                "dataset": {"sha256": WORKLOAD.sha256_file(dataset)},
                "contract": contract,
            }
            self.assertEqual(
                WORKLOAD.verify_manifest(
                    manifest, dataset=dataset, expected=contract
                ),
                [],
            )
            dataset.write_text(json.dumps({"conversations": [[], [], []]}))
            self.assertIn(
                "dataset SHA-256 differs from the workload manifest",
                WORKLOAD.verify_manifest(
                    manifest, dataset=dataset, expected=contract
                ),
            )


if __name__ == "__main__":
    unittest.main()
