"""CPU-only checks: python3 -m unittest discover -s recipes/deepseek-v41/launcher."""

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = load("launch")
probe = load("probe_engram")
car_probe = load("probe_custom_all_reduce")


class TestLaunch(unittest.TestCase):
    def test_split_graphs_use_phase_specific_backends(self):
        plan = launch.build_plan(
            self.args("--cpu-engram-staging", "--no-dspark", "--split-cuda-graphs")
        )
        for role, worker in zip(("prefill", "decode"), plan["workers"]):
            command = worker["command"]
            config = json.loads(command[command.index("--cuda-graph-config") + 1])
            self.assertEqual(config[role]["backend"], "breakable")
            self.assertEqual(
                config["decode" if role == "prefill" else "prefill"]["backend"],
                "disabled",
            )
            self.assertNotIn("--disable-decode-cuda-graph", command)
            self.assertNotIn("--disable-prefill-cuda-graph", command)
            self.assertEqual(
                "--enable-decoder-swa-bounded-replay" in command, role == "decode"
            )
            self.assertEqual(
                "--no-enable-decoder-swa-bounded-replay" in command, role == "prefill"
            )

    def test_bounded_prefill_graph_plan(self):
        plan = launch.build_plan(
            self.args(
                "--cpu-engram-staging",
                "--split-cuda-graphs",
                "--prefill-bounded-replay",
            )
        )
        for worker in plan["workers"]:
            self.assertIn("--enable-decoder-swa-bounded-replay", worker["command"])
        with self.assertRaisesRegex(ValueError, "requires --split-cuda-graphs"):
            launch.build_plan(self.args("--prefill-bounded-replay"))

    def test_prefill_tail_pp_plan_preserves_normal_decode_handoff(self):
        args = self.args(
            "--cpu-engram-staging",
            "--split-cuda-graphs",
            "--prefill-bounded-replay",
            "--prefill-pp",
            "2",
        )
        plan = launch.build_plan(args)
        self.assertEqual(
            plan["workers"][0]["env"]["SGLANG_PP_LAYER_PARTITION"], "14,26"
        )
        for worker in plan["workers"]:
            self.assertNotIn("SGLANG_OPT_DSV41_CED_SPLIT", worker["env"])
            self.assertEqual(worker["env"]["SGLANG_OPT_DSV41_PREFILL_TAIL"], "1")
            self.assertIn("--speculative-algorithm", worker["command"])
        args.ced_split = True
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            launch.build_plan(args)

    def test_cleanup_only_selects_owned_gpu_pids_and_direct_children(self):
        parent = Mock(pid=11)
        parent.poll.return_value = None
        with (
            patch.object(
                launch.subprocess,
                "check_output",
                side_effect=[
                    "0, GPU-a\n4, GPU-b\n",
                    "21, GPU-a\n22, GPU-b\n23, GPU-a\n10436, GPU-a\n",
                ],
            ),
            patch.object(
                launch.os, "getsid", side_effect=lambda pid: 11 if pid == 21 else 99
            ),
        ):
            self.assertEqual(
                launch.owned_cleanup_pids([("prefill", parent)], [0], 10436),
                {11, 21},
            )

    def test_split_graphs_cover_larger_decode_limits(self):
        for limit in (32, 48, 64):
            plan = launch.build_plan(
                self.args(
                    "--cpu-engram-staging",
                    "--no-dspark",
                    "--split-cuda-graphs",
                    "--decode-requests",
                    str(limit),
                )
            )
            command = plan["workers"][1]["command"]
            config = json.loads(command[command.index("--cuda-graph-config") + 1])
            buckets = config["decode"]["bs"]
            self.assertEqual(max(buckets), limit)
            self.assertEqual(buckets[:5], [1, 2, 4, 8, 16])
            self.assertEqual(buckets, sorted(set(buckets)))

    def test_split_graphs_reject_unqualified_modes(self):
        with self.assertRaisesRegex(ValueError, "require CPU staging"):
            launch.build_plan(self.args("--split-cuda-graphs"))

    def test_split_graphs_enable_bundled_dspark_on_both_roles(self):
        plan = launch.build_plan(
            self.args(
                "--cpu-engram-staging", "--split-cuda-graphs", "--decode-requests", "64"
            )
        )
        for worker in plan["workers"]:
            command = worker["command"]
            self.assertEqual(
                command[command.index("--speculative-algorithm") + 1], "DSPARK"
            )
            self.assertEqual(
                command[command.index("--speculative-dspark-block-size") + 1], "5"
            )
            self.assertNotIn("--speculative-draft-model-path", command)
            self.assertEqual(worker["env"]["MC_INTRANODE_NVLINK"], "true")
        command = plan["workers"][1]["command"]
        config = json.loads(command[command.index("--cuda-graph-config") + 1])
        self.assertEqual(config["decode"]["backend"], "breakable")
        self.assertEqual(config["decode"]["bs"][-1], 64)

    def test_router_readiness_retries_until_workers_registered(self):
        from urllib.error import HTTPError

        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.status = 200
        with (
            patch.object(
                launch.urllib.request,
                "urlopen",
                side_effect=[
                    HTTPError("http://router/readiness", 503, "not ready", {}, None),
                    response,
                ],
            ) as get,
            patch.object(launch.time, "sleep"),
        ):
            launch.wait_ready("http://router", [], 10, path="/readiness")
        self.assertEqual(get.call_count, 2)
        get.assert_called_with("http://router/readiness", timeout=3)

    def test_removed_indexer_flags_fail_with_their_replacement(self):
        for flag, reason in (
            ("--prefill-hoist-per-batch", "not ported to cc-main"),
            ("--prefill-graph-eager-indexer", "--prefill-graph-max-seq-len"),
        ):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, reason):
                launch.validate(self.args(flag))

    def test_prefill_graph_max_seq_len_lifts_the_cap_on_prefill_only(self):
        eager_env = "SGLANG_OPT_DSV41_PREFILL_GRAPH_EAGER_INDEXER"
        base = ("--cpu-engram-staging", "--split-cuda-graphs")

        def graph_configs(*extra):
            prefill, decode = launch.build_plan(self.args(*base, *extra))["workers"]
            graphs = []
            for worker in (prefill, decode):
                command = worker["command"]
                graphs.append(
                    json.loads(command[command.index("--cuda-graph-config") + 1])
                )
            return prefill, decode, graphs

        prefill, decode, (prefill_graphs, _) = graph_configs()
        self.assertNotIn("max_seq_len", prefill_graphs["prefill"])
        self.assertNotIn(eager_env, prefill["env"])

        prefill, decode, (prefill_graphs, decode_graphs) = graph_configs(
            "--prefill-graph-max-seq-len", "131072"
        )
        self.assertEqual(prefill_graphs["prefill"]["max_seq_len"], 131072)
        self.assertEqual(prefill["env"][eager_env], "1")
        self.assertEqual(decode_graphs["prefill"], {"backend": "disabled"})
        self.assertNotIn(eager_env, decode["env"])

        # Within the captured indexer's window the cap needs no eager indexer.
        prefill, _, (prefill_graphs, _) = graph_configs(
            "--prefill-graph-max-seq-len", "8192"
        )
        self.assertEqual(prefill_graphs["prefill"]["max_seq_len"], 8192)
        self.assertNotIn(eager_env, prefill["env"])

    def test_prefill_graph_max_seq_len_defaults_to_the_context_length(self):
        eager_env = "SGLANG_OPT_DSV41_PREFILL_GRAPH_EAGER_INDEXER"

        def prefill_graph(*extra):
            prefill, _ = launch.build_plan(
                self.args("--cpu-engram-staging", "--split-cuda-graphs", *extra)
            )["workers"]
            command = prefill["command"]
            config = json.loads(command[command.index("--cuda-graph-config") + 1])
            return config["prefill"], prefill["env"]

        long_context = ("--context-length", "1048576", "--max-total-tokens", "1572864")
        graph, env = prefill_graph(*long_context)
        self.assertEqual(graph["max_seq_len"], 1048576)
        self.assertEqual(env[eager_env], "1")

        # An explicit cap still wins, including the server's own.
        graph, env = prefill_graph(*long_context, "--prefill-graph-max-seq-len", "16384")
        self.assertEqual(graph["max_seq_len"], 16384)
        self.assertNotIn(eager_env, env)

        # Bounded replay is not validated past the captured indexer's window.
        graph, env = prefill_graph(*long_context, "--prefill-bounded-replay")
        self.assertNotIn("max_seq_len", graph)
        self.assertNotIn(eager_env, env)

    def test_prefill_graph_max_seq_len_rejects_unsupported_plans(self):
        for extra, reason in (
            (("--prefill-graph-max-seq-len", "32768"), "requires --split-cuda-graphs"),
            (
                (
                    "--split-cuda-graphs",
                    "--prefill-graph-max-seq-len",
                    "32768",
                    "--prefill-bounded-replay",
                ),
                "not validated with --prefill-bounded-replay",
            ),
            (
                ("--split-cuda-graphs", "--prefill-graph-max-seq-len", "0"),
                "must be positive",
            ),
        ):
            with (
                self.subTest(extra=extra),
                self.assertRaisesRegex(ValueError, reason),
            ):
                launch.validate(self.args("--cpu-engram-staging", *extra))

    def args(self, *extra):
        return launch.parser().parse_args(["--model", "/missing/checkpoint", *extra])

    def test_staging_disables_graphs_in_both_workers(self):
        plan = launch.build_plan(self.args("--cpu-engram-staging"))
        for worker in plan["workers"]:
            self.assertEqual(worker["env"]["SGLANG_DSV41_ENGRAM_CPU_STAGING"], "1")
            self.assertIn("--disable-decode-cuda-graph", worker["command"])
            self.assertIn("--disable-prefill-cuda-graph", worker["command"])
            self.assertNotIn("--cuda-graph-max-bs-decode", worker["command"])

    def test_residency_credit_requires_readonly_locked_matching_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            shard = model / "table.safetensors"
            shard.write_bytes(bytes(64))
            stat = shard.stat()
            index = json.dumps(
                {"weight_map": {"layers.1.engram.embed.weight": shard.name}}
            )
            device = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}"
            for permissions, flags, inode, valid in (
                ("r--s", "rd lo", stat.st_ino, True),
                ("rw-p", "rd wr lo", stat.st_ino, False),
                ("r--s", "rd", stat.st_ino, False),
                ("r--s", "rd lo", stat.st_ino + 1, False),
            ):
                smaps = (
                    f"1000-2000 {permissions} 00000000 {device} {inode} {shard}\n"
                    f"Size: 4 kB\nRss: 4 kB\nVmFlags: {flags}\n"
                )
                with patch.object(Path, "read_text", side_effect=[index, smaps]):
                    if valid:
                        self.assertEqual(launch.resident_engram_bytes(model, 123), 64)
                    else:
                        with self.assertRaises(ValueError):
                            launch.resident_engram_bytes(model, 123)

    def test_parallel_loading_requires_shared_staging(self):
        with self.assertRaises(ValueError):
            launch.validate(self.args("--parallel-load"))
        launch.validate(self.args("--parallel-load", "--cpu-engram-staging"))

    def test_dry_run_needs_no_model_cuda_or_serving_dependencies(self):
        completed = subprocess.run(
            [
                os.sys.executable,
                str(Path(launch.__file__)),
                "--model",
                "/missing/checkpoint",
                "--dry-run",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        plan = json.loads(completed.stdout)
        self.assertEqual(plan["status"], "unqualified")
        self.assertEqual(plan["physical_gpus"], [0, 1, 2, 3])

    def test_symmetric_transfer_layout_and_disjoint_devices(self):
        plan = launch.build_plan(self.args("--gpus", "4,5,6,7"))
        prefill, decode = plan["workers"]
        self.assertEqual(prefill["env"]["CUDA_VISIBLE_DEVICES"], "4,5")
        self.assertEqual(decode["env"]["CUDA_VISIBLE_DEVICES"], "6,7")
        for flag in (
            "--tp-size",
            "--ep-size",
            "--dp-size",
            "--page-size",
            "--kv-cache-dtype",
            "--speculative-dspark-block-size",
            "--disaggregation-transfer-backend",
        ):
            values = [
                w["command"][w["command"].index(flag) + 1] for w in plan["workers"]
            ]
            self.assertEqual(values[0], values[1], flag)
        self.assertNotEqual(
            prefill["env"]["TRITON_CACHE_DIR"], decode["env"]["TRITON_CACHE_DIR"]
        )
        for w in plan["workers"]:
            self.assertIn("--disable-prefill-cuda-graph", w["command"])
            self.assertNotIn("--enable-encoder-swa-bounded-replay", w["command"])
            self.assertNotIn("--enable-dp-attention", w["command"])

    def test_three_prefill_one_decode_plan(self):
        plan = launch.build_plan(
            self.args(
                "--num-prefill",
                "3",
                "--gpus",
                "7,6,5,4,3,2,1,0",
                "--prefill-swa-prefix-tails",
                "8",
                "--decode-max-total-tokens",
                "131072",
            )
        )
        workers = plan["workers"]
        self.assertEqual(
            [w["name"] for w in workers], ["prefill0", "prefill1", "prefill2", "decode"]
        )
        self.assertEqual(
            [w["env"]["CUDA_VISIBLE_DEVICES"] for w in workers],
            ["7,6", "5,4", "3,2", "1,0"],
        )

        def flag(worker, name):
            command = worker["command"]
            return command[command.index(name) + 1] if name in command else None

        http = [int(w["url"].rsplit(":", 1)[1]) for w in workers]
        nccl = [int(flag(w, "--nccl-port")) for w in workers]
        bootstrap = [int(flag(w, "--disaggregation-bootstrap-port")) for w in workers]
        self.assertEqual([int(flag(w, "--port")) for w in workers], http)
        self.assertEqual([w.get("bootstrap_port") for w in workers[:3]], bootstrap[:3])
        ports = [8030, 29000, *http, *nccl, *bootstrap[:3]]
        self.assertEqual(len(set(ports)), len(ports), ports)
        # Decode only uses the flag as a default prefill target; it binds nothing.
        self.assertIn(bootstrap[3], bootstrap[:3])
        for worker in workers[:3]:
            self.assertEqual(flag(worker, "--swa-prefix-tails"), "8")
            self.assertEqual(flag(worker, "--max-total-tokens"), "65536")
        self.assertIsNone(flag(workers[3], "--swa-prefix-tails"))
        self.assertEqual(flag(workers[3], "--max-total-tokens"), "131072")
        router = plan["router"]["command"]
        prefill_pairs = [
            (router[i + 1], int(router[i + 2]))
            for i, arg in enumerate(router)
            if arg == "--prefill"
        ]
        self.assertEqual(
            prefill_pairs,
            [(w["url"], p) for w, p in zip(workers[:3], bootstrap[:3])],
        )
        self.assertEqual(router[router.index("--decode") + 1], workers[3]["url"])
        self.assertEqual(router[router.index("--prefill-policy") + 1], "manual")

    def test_n_prefill_m_decode_plan_with_per_role_tp(self):
        plan = launch.build_plan(
            self.args(
                "--num-prefill",
                "1",
                "--prefill-tp",
                "4",
                "--num-decode",
                "2",
                "--gpus",
                "7,6,5,4,3,2,1,0",
            )
        )
        workers = plan["workers"]
        self.assertEqual(
            [w["name"] for w in workers], ["prefill", "decode0", "decode1"]
        )
        self.assertEqual(
            [w["env"]["CUDA_VISIBLE_DEVICES"] for w in workers],
            ["7,6,5,4", "3,2", "1,0"],
        )

        def flag(worker, name):
            return worker["command"][worker["command"].index(name) + 1]

        self.assertEqual([flag(w, "--tp-size") for w in workers], ["4", "2", "2"])
        self.assertEqual([flag(w, "--ep-size") for w in workers], ["4", "2", "2"])
        self.assertEqual(
            [flag(w, "--disaggregation-mode") for w in workers],
            ["prefill", "decode", "decode"],
        )
        ports = [int(flag(w, "--port")) for w in workers]
        nccl = [int(flag(w, "--nccl-port")) for w in workers]
        self.assertEqual(len(set(ports + nccl)), 6)
        self.assertTrue(plan["topology"].startswith("1x(TP4 prefill) + 2x(TP2 decode)"))
        router = plan["router"]["command"]
        decodes = [router[i + 1] for i, a in enumerate(router) if a == "--decode"]
        self.assertEqual(decodes, [w["url"] for w in workers[1:]])
        # One prefill needs no prefill policy; several decodes stay sticky per key.
        self.assertNotIn("--prefill-policy", router)
        self.assertEqual(router[router.index("--decode-policy") + 1], "manual")
        self.assertEqual(router[router.index("--assignment-mode") + 1], "min_group")
        for extra in (
            ("--num-decode", "0"),
            ("--gpus", "0,1,2,3,4,5,6"),
            ("--prefill-pp", "2"),
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                launch.validate(
                    self.args("--prefill-tp", "4", "--num-decode", "2", *extra)
                )

    def test_memory_profile_sizes_pools_and_reports_the_estimate(self):
        def args(*extra):
            return self.args(
                "--cpu-engram-staging",
                "--split-cuda-graphs",
                "--chunk-size",
                "12288",
                "--prefill-requests",
                "16",
                "--prefill-swa-prefix-tails",
                "256",
                "--decode-requests",
                "64",
                "--context-length",
                "131072",
                "--mem-fraction",
                "0.92",
                "--max-total-tokens",
                "auto",
                "--decode-max-total-tokens",
                "auto",
                *extra,
            )

        # The B200 profile reproduces the measured pair: ~10 GiB free on prefill
        # (9.67 measured at a 1.57M pool) and the decode cap set by --mem-fraction.
        plan = launch.build_plan(args("--memory-profile", "B200"))
        memory = plan["memory"]["per_rank"]
        self.assertEqual(memory["prefill"]["pool_tokens"], 1638400)
        self.assertEqual(memory["decode"]["pool_tokens"], 9568256)
        command = plan["workers"][0]["command"]
        self.assertEqual(command[command.index("--max-total-tokens") + 1], "1638400")
        b300 = launch.build_plan(
            args("--memory-profile", "B300", "--context-length", "1048576")
        )["memory"]["per_rank"]
        self.assertGreater(b300["decode"]["contexts_resident"], 32)
        with self.assertRaisesRegex(ValueError, "needs --memory-profile"):
            launch.build_plan(args())
        # Without a profile the plan carries no estimate (today's plans unchanged).
        self.assertNotIn("memory", launch.build_plan(self.args()))

    def test_two_ced_pp2_prefills_two_tp2_decodes_with_stage_memory(self):
        args = self.args(
            "--cpu-engram-staging",
            "--split-cuda-graphs",
            "--ced-split",
            "--prefill-pp",
            "2",
            "--num-prefill",
            "2",
            "--num-decode",
            "2",
            "--chunk-size",
            "12288",
            "--prefill-requests",
            "16",
            "--prefill-swa-prefix-tails",
            "256",
            "--decode-requests",
            "64",
            "--context-length",
            "1048576",
            "--mem-fraction",
            "0.92",
            "--memory-profile",
            "B300",
            "--max-total-tokens",
            "auto",
            "--decode-max-total-tokens",
            "auto",
            "--prefill-hicache-size-gib",
            "32",
        )
        plan = launch.build_plan(args)
        workers = plan["workers"]
        self.assertEqual(
            [w["env"]["CUDA_VISIBLE_DEVICES"] for w in workers],
            ["0,1", "2,3", "4,5", "6,7"],
        )
        self.assertTrue(
            plan["topology"].startswith("2x(PP2xTP1 CED prefill) + 2x(TP2 decode)")
        )

        def flag(worker, name):
            command = worker["command"]
            return command[command.index(name) + 1] if name in command else None

        for prefill in workers[:2]:
            self.assertEqual(flag(prefill, "--pp-size"), "2")
            self.assertEqual(flag(prefill, "--tp-size"), "1")
            self.assertEqual(prefill["env"]["SGLANG_PP_LAYER_PARTITION"], "8,13")
            self.assertIsNone(flag(prefill, "--speculative-algorithm"))
        for decode in workers[2:]:
            self.assertEqual(flag(decode, "--tp-size"), "2")
            self.assertEqual(flag(decode, "--speculative-algorithm"), "DSPARK")
        for worker in workers:
            self.assertEqual(worker["env"]["SGLANG_OPT_DSV41_CED_SPLIT"], "1")
        router = plan["router"]["command"]
        self.assertEqual(router[router.index("--prefill-policy") + 1], "manual")
        self.assertEqual(router[router.index("--decode-policy") + 1], "manual")
        # Both encoder stages share one pool; the heavier 8-20 stage sets it.
        prefill = plan["memory"]["per_rank"]["prefill"]
        stages = prefill["stages"]
        self.assertEqual([s["layers"] for s in stages], ["0-7", "8-20"])
        self.assertEqual(prefill["max_pool_tokens"], stages[1]["max_pool_tokens"])
        self.assertLess(stages[1]["free_gib"], stages[0]["free_gib"])
        self.assertEqual(
            flag(workers[0], "--max-total-tokens"), str(prefill["max_pool_tokens"])
        )
        # 32 GiB per stage rank, two stages, two workers.
        self.assertEqual(plan["memory"]["hicache_host_gib_all_prefill"], 128.0)
        args.prefill_pp_partition = "14,7"
        with self.assertRaisesRegex(ValueError, "default 8,13 partition"):
            launch.pool_estimate(args, "prefill", 2)

    def test_hicache_size_replaces_the_ratio_and_counts_against_host_ram(self):
        plan = launch.build_plan(self.args("--prefill-hicache-size-gib", "96"))
        command = plan["workers"][0]["command"]
        self.assertEqual(command[command.index("--hicache-size") + 1], "96")
        self.assertNotIn("--hicache-ratio", command)
        self.assertNotIn("--hicache-size", plan["workers"][1]["command"])
        args = self.args("--num-prefill", "3", "--prefill-hicache-size-gib", "96")
        self.assertEqual(
            3 * launch.hicache_host_bytes(args, 2, args.max_total_tokens),
            3 * 96 * launch.GIB * 2,
        )
        with self.assertRaisesRegex(ValueError, "exclusive"):
            launch.validate(
                self.args(
                    "--prefill-hicache-size-gib", "96", "--prefill-hicache-ratio", "4"
                )
            )

    def test_multi_prefill_rejects_colliding_ports_and_short_gpu_lists(self):
        cases = [
            ("--gpus", "0,1,2,3"),
            ("--bootstrap-port", "8033"),
            ("--router-metrics-port", "29503"),
            ("--port", "8996"),
            ("--num-prefill", "0"),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                launch.validate(self.args("--num-prefill", "3", *args))

    def test_target_only_control_disables_speculation_on_both_sides(self):
        plan = launch.build_plan(self.args("--no-dspark"))
        for worker in plan["workers"]:
            self.assertNotIn("--speculative-algorithm", worker["command"])

    def test_custom_all_reduce_is_opt_in_and_changes_only_the_nccl_pin(self):
        default = launch.build_plan(self.args("--num-prefill", "2"))
        enabled = launch.build_plan(
            self.args("--num-prefill", "2", "--custom-all-reduce", "on")
        )
        self.assertEqual(default["collectives"]["custom_all_reduce"], "off")
        self.assertEqual(enabled["collectives"]["custom_all_reduce"], "on")
        self.assertEqual(len(enabled["workers"]), 3)
        for off, on in zip(default["workers"], enabled["workers"]):
            self.assertIn("--disable-custom-all-reduce", off["command"])
            self.assertNotIn("--disable-custom-all-reduce", on["command"])
            self.assertEqual(
                [x for x in off["command"] if x != "--disable-custom-all-reduce"],
                on["command"],
            )
            self.assertIn(
                "--enforce-disable-flashinfer-allreduce-fusion", on["command"]
            )
            self.assertEqual(off["env"], on["env"])

    def test_dry_run_prints_the_resolved_collective_plan(self):
        for mode in ("off", "on"):
            completed = subprocess.run(
                [
                    os.sys.executable,
                    str(Path(launch.__file__)),
                    "--model",
                    "/missing/checkpoint",
                    "--dry-run",
                    "--custom-all-reduce",
                    mode,
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            plan = json.loads(completed.stdout)
            self.assertEqual(plan["collectives"]["custom_all_reduce"], mode)
            for worker in plan["workers"]:
                self.assertEqual(
                    "--disable-custom-all-reduce" in worker["command"], mode == "off"
                )

    def test_custom_all_reduce_probe_skips_workers_without_a_tp_group(self):
        plan = launch.build_plan(
            self.args("--custom-all-reduce", "on", "--num-prefill", "2")
        )
        self.assertEqual(
            launch.custom_all_reduce_probe_gpus(plan), ["0,1", "2,3", "4,5"]
        )
        # A PP2 prefill worker runs TP1 per stage, so it has no TP all-reduce.
        plan = launch.build_plan(
            self.args("--custom-all-reduce", "on", "--prefill-pp", "2")
        )
        self.assertEqual(launch.custom_all_reduce_probe_gpus(plan), ["2,3"])

    def test_custom_all_reduce_probe_failure_fails_preflight_with_its_verdict(self):
        args = self.args("--custom-all-reduce", "on", "--output-dir", "/tmp/unused")
        plan = launch.build_plan(args)
        silent_fallback = Mock(
            returncode=2,
            stdout='{"reports": {}}\nNOT INITIALIZED: ca_comm is None\n',
            stderr="",
        )
        with patch.object(launch.subprocess, "run", return_value=silent_fallback):
            with self.assertRaisesRegex(RuntimeError, "NOT INITIALIZED"):
                launch.probe_custom_all_reduce(args, plan, "0,1")

    def test_probe_verdict_tells_silent_fallback_from_mismatch_and_hang(self):
        up = dict(initialized=True, warnings=[], summary="")
        down = dict(initialized=False, warnings=["Setup failed"], summary="NCCL")
        ok = dict(stage="done", custom_all_reduce=up, correctness=dict(all_match=True))
        fallback = dict(stage="group_ready", custom_all_reduce=down)
        mismatch = dict(ok, correctness=dict(all_match=False))
        cases = [
            ({0: ok, 1: ok}, False, car_probe.PASS),
            ({0: ok, 1: fallback}, False, car_probe.NOT_INITIALIZED),
            ({0: ok, 1: mismatch}, False, car_probe.FAIL),
            ({0: ok}, False, car_probe.FAIL),
            ({0: dict(stage="group_init"), 1: ok}, True, car_probe.HANG),
        ]
        for reports, timed_out, expected in cases:
            with self.subTest(expected=expected):
                code, _ = car_probe.verdict(reports, timed_out=timed_out, world_size=2)
                self.assertEqual(code, expected)

    def test_silent_nccl_fallback_in_a_worker_log_fails_the_launch(self):
        # The launcher reads server log lines, so a reworded line must fail here.
        distributed = launch.ROOT / "python/sglang/srt/distributed"
        self.assertIn(
            launch.CUSTOM_AR_READY_LINE,
            (distributed / "device_communicators/custom_all_reduce_v2.py").read_text(),
        )
        self.assertIn(
            launch.CUSTOM_AR_FAILED_LINE,
            (distributed / "parallel_state.py").read_text(),
        )
        ready = "INFO All Reduce config: symmetric_memory = 80.02 MB\n"
        failed = "WARNING Setup Custom allreduce failed with boom\n"
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "decode.log"
            log.write_text(ready)
            launch.require_custom_all_reduce(log)
            for text, pattern in (
                (failed, "setup failed"),
                (ready + failed, "setup failed"),
                ("INFO unrelated\n", "never initialized"),
            ):
                log.write_text(text)
                with (
                    self.subTest(text=text),
                    self.assertRaisesRegex(RuntimeError, pattern),
                ):
                    launch.require_custom_all_reduce(log)

    def test_invalid_topologies_and_memory_limits(self):
        cases = [
            ("--gpus", "0,1,1,2"),
            ("--gpus", "0,1"),
            ("--gpus", "0,1,2,-1"),
            ("--port", "8998"),
            ("--port", "65535"),
            ("--mem-fraction", "1"),
            ("--chunk-size", "0"),
            ("--context-length", "100000"),
            ("--dspark-block-size", "0"),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                launch.validate(self.args(*args))

    def test_transports_do_not_inherit_previous_experiment(self):
        with patch.dict(
            os.environ,
            {
                "MC_FORCE_MNNVL": "True",
                "NCCL_ALGO": "NVLS",
                "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "0",
                "CUDA_VISIBLE_DEVICES": "7",
                "PYTHONPATH": "/wrong",
            },
        ):
            for transport in ("tcp", "rdma", "intra-node-nvlink"):
                plan = launch.build_plan(self.args("--transport", transport))
                env = launch.clean_environment(plan["workers"][0]["env"])
                self.assertNotIn("MC_FORCE_MNNVL", env)
                self.assertNotIn("NCCL_ALGO", env)
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
                self.assertEqual(env["SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE"], "1")
                self.assertEqual(
                    env.get("MC_FORCE_TCP"), "1" if transport == "tcp" else None
                )
                self.assertEqual(
                    env.get("MC_INTRANODE_NVLINK"),
                    "true" if transport == "intra-node-nvlink" else None,
                )
                command = plan["workers"][0]["command"]
                backend = command[
                    command.index("--disaggregation-transfer-backend") + 1
                ]
                self.assertEqual(backend, "mooncake")

    def test_schedule_policy_is_prefill_only(self):
        plan = launch.build_plan(self.args("--prefill-schedule-policy", "hrrn"))
        for worker in plan["workers"]:
            command = worker["command"]
            role = command[command.index("--disaggregation-mode") + 1]
            if role == "prefill":
                self.assertEqual(
                    command[command.index("--schedule-policy") + 1], "hrrn"
                )
            else:
                self.assertNotIn("--schedule-policy", command)

    def test_request_time_stats_apply_to_both_roles(self):
        for flag, expected in ((), False), (("--request-time-stats",), True):
            for worker in launch.build_plan(self.args(*flag))["workers"]:
                self.assertEqual(
                    "--enable-request-time-stats-logging" in worker["command"],
                    expected,
                )

    def test_router_drops_idle_connections_before_servers_close_them(self):
        plan = launch.build_plan(self.args())
        for worker in plan["workers"]:
            self.assertGreater(
                int(worker["env"]["SGLANG_TIMEOUT_KEEP_ALIVE"]),
                launch.ROUTER_POOL_IDLE_S,
            )

    def test_role_env_accepts_nccl_keys_for_that_role_only(self):
        plan = launch.build_plan(
            self.args("--prefill-env", "NCCL_PROTO=LL128", "--decode-env", "SGLANG_X=1")
        )
        prefill, decode = plan["workers"]
        self.assertEqual(prefill["env"]["NCCL_PROTO"], "LL128")
        self.assertNotIn("NCCL_PROTO", decode["env"])
        self.assertEqual(decode["env"]["SGLANG_X"], "1")
        for bad in ("PATH=/x", "NCCL_PROTO", "MC_FORCE_TCP=1"):
            with self.assertRaises(ValueError):
                launch.build_plan(self.args("--prefill-env", bad))

    def test_decode_radix_cache_is_decode_only(self):
        flag = "--disaggregation-decode-enable-radix-cache"
        prefill, decode = launch.build_plan(self.args())["workers"]
        self.assertNotIn(flag, decode["command"])
        prefill, decode = launch.build_plan(self.args("--decode-radix-cache"))[
            "workers"
        ]
        self.assertIn(flag, decode["command"])
        self.assertNotIn(flag, prefill["command"])

    def test_decode_graph_extra_bs_and_polling_interval_are_decode_only(self):
        plan = launch.build_plan(
            self.args(
                "--cpu-engram-staging",
                "--split-cuda-graphs",
                "--decode-requests",
                "64",
                "--decode-graph-extra-bs",
                "10,12,14,100",
                "--decode-polling-interval",
                "2",
            )
        )
        prefill, decode = plan["workers"]
        graphs = json.loads(
            decode["command"][decode["command"].index("--cuda-graph-config") + 1]
        )
        self.assertEqual(graphs["decode"]["bs"], [1, 2, 4, 8, 10, 12, 14, 16, 32, 64])
        flag = "--disaggregation-decode-polling-interval"
        self.assertEqual(decode["command"][decode["command"].index(flag) + 1], "2")
        self.assertNotIn(flag, prefill["command"])
        for bad in (
            ("--decode-polling-interval", "0"),
            ("--decode-graph-extra-bs", "0"),
            ("--decode-graph-extra-bs", "12"),  # needs split graphs
        ):
            with self.assertRaises(ValueError):
                launch.build_plan(self.args(*bad))

    def test_tokenizer_workers_apply_to_both_roles(self):
        flag = "--tokenizer-worker-num"
        for worker in launch.build_plan(self.args())["workers"]:
            self.assertNotIn(flag, worker["command"])
        plan = launch.build_plan(self.args("--tokenizer-workers", "4"))
        for worker in plan["workers"]:
            command = worker["command"]
            self.assertEqual(command[command.index(flag) + 1], "4")
        with self.assertRaises(ValueError):
            launch.build_plan(self.args("--tokenizer-workers", "0"))

    def test_hicache_is_prefill_only(self):
        plan = launch.build_plan(self.args("--prefill-hicache-ratio", "2"))
        for worker in plan["workers"]:
            command = worker["command"]
            role = command[command.index("--disaggregation-mode") + 1]
            enabled = "--enable-hierarchical-cache" in command
            self.assertEqual(enabled, role == "prefill")
            self.assertEqual(
                worker["env"].get("SGLANG_ENABLE_HICACHE_SWA_TOMBSTONE_BACKUP"),
                "0" if enabled else None,
            )
            if enabled:
                self.assertEqual(command[command.index("--hicache-ratio") + 1], "2.0")
                self.assertEqual(
                    command[command.index("--hicache-io-backend") + 1], "direct"
                )
                self.assertEqual(
                    command[command.index("--hicache-write-policy") + 1], "write_back"
                )
        with self.assertRaises(ValueError):
            launch.validate(self.args("--prefill-hicache-ratio", "1"))

    def test_branch_pins_and_candidate_indexer_are_explicit(self):
        self.assertEqual(
            set(launch.required_versions()),
            {"torch", "transformers", "flashinfer-python", "sglang-kernel"},
        )
        plan = launch.build_plan(self.args("--candidate-indexer", "deepgemm"))
        for worker in plan["workers"]:
            self.assertEqual(
                worker["env"]["SGLANG_DSV41_DEEP_GEMM_CANDIDATE_INDEXER"], "1"
            )

    def test_checkpoint_budget_counts_two_copies_not_four(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            config = dict(
                model_type="deepseek_v41",
                text_config=dict(
                    engram_num_embeddings=[100, 200],
                    engram_layer_ids=[0, 14],
                    engram_head_dim=256,
                    num_attention_heads=64,
                    o_groups=8,
                    n_routed_experts=256,
                ),
            )
            (path / "config.json").write_text(json.dumps(config))
            (path / "model-00001.safetensors").write_bytes(b"test")
            result = launch.checkpoint_budget(path)
            self.assertEqual(result["engram_bytes_per_instance"], 300 * (256 + 8))
            self.assertEqual(result["engram_bytes_pd_pair"], 2 * 300 * (256 + 8))
            # nvidia/DeepSeek-V4.1-Flash-NVFP4 packs two e2m1 values per byte.
            fp4_config = dict(
                config, quantization_config=dict(engram_embed_quant="mxfp4")
            )
            (path / "config.json").write_text(json.dumps(fp4_config))
            result = launch.checkpoint_budget(path)
            self.assertEqual(result["engram_bytes_per_instance"], 300 * (128 + 8))
            (path / "config.json").write_text(json.dumps(config))
            index = dict(
                metadata={"total_size": 100},
                weight_map={
                    "a": "model-00001.safetensors",
                    "b": "model-00002.safetensors",
                },
            )
            (path / "model.safetensors.index.json").write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                launch.checkpoint_budget(path)
            (path / "model-00002.safetensors").write_bytes(b"test")
            with self.assertRaisesRegex(ValueError, "truncated"):
                launch.checkpoint_budget(path)
            config["text_config"]["o_groups"] = 1
            (path / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "TP2/EP2"):
                launch.checkpoint_budget(path)
            # 6 splits two ways but not four; TP4 is checked before the shards.
            config["text_config"]["o_groups"] = 6
            (path / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "truncated"):
                launch.checkpoint_budget(path)
            with self.assertRaisesRegex(ValueError, "TP4/EP4"):
                launch.checkpoint_budget(path, (2, 4))

    def test_host_registration_failure_cannot_be_bypassed_with_unpinned_memory(self):
        self.assertIn(
            "unsupported",
            probe.host_access_error(
                dict(
                    host_register_supported=0,
                    can_map_host_memory=1,
                    pageable_memory_access=0,
                )
            ),
        )
        self.assertIsNone(
            probe.host_access_error(
                dict(
                    host_register_supported=1,
                    can_map_host_memory=1,
                    pageable_memory_access=0,
                )
            )
        )

    def test_compaction_counts_only_normal_zone_2mib_blocks(self):
        # /proc/buddyinfo when CC cuInit failed, and after the workers exited.
        failed = (
            "Node 0, zone      DMA      1      2      1      0      1      1      1      1      2      2      1\n"
            "Node 0, zone    DMA32      9     12     17     10     12     11     10      7      8      2    245\n"
            "Node 0, zone   Normal  31325  11837   9059  13325  22573    530    133     75     33      0      0\n"
        )
        recovered = failed.replace("33      0      0", "33    872    225")
        self.assertEqual(launch.free_contiguous_blocks(failed), 0)
        self.assertEqual(launch.free_contiguous_blocks(recovered), 872 + 2 * 225)
        with (
            patch.object(launch.subprocess, "run") as run,
            patch.object(launch.Path, "read_text", return_value=failed),
            patch.object(launch.os, "geteuid", return_value=1000),
        ):
            with self.assertRaisesRegex(RuntimeError, "only 0 free 2 MiB"):
                launch.compact_host_memory()
            self.assertEqual(run.call_args.args[0][:2], ["sudo", "-n"])
        # The container runs as root and has no sudo binary.
        with (
            patch.object(launch.subprocess, "run") as run,
            patch.object(launch.Path, "read_text", return_value=recovered),
            patch.object(launch.os, "geteuid", return_value=0),
        ):
            launch.compact_host_memory()
            self.assertEqual(run.call_args.args[0][:2], ["sh", "-c"])

    def test_readiness_checks_all_owned_processes_before_http(self):
        process = Mock()
        process.poll.return_value = 1
        process.returncode = 1
        with patch.object(launch.urllib.request, "urlopen") as get:
            with self.assertRaisesRegex(RuntimeError, "prefill exited"):
                launch.wait_ready("http://localhost:8032", [("prefill", process)], 5)
            get.assert_not_called()

    def test_failed_preflight_never_starts_workers(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            with (
                patch.object(
                    launch,
                    "preflight",
                    return_value={
                        "passed": False,
                        "errors": ["unsupported host access"],
                    },
                ),
                patch.object(launch, "serve") as serve,
            ):
                self.assertEqual(
                    launch.main(["--model", "/missing", "--output-dir", str(out)]), 1
                )
                serve.assert_not_called()
            self.assertFalse(json.loads((out / "preflight.json").read_text())["passed"])

    def test_existing_artifacts_are_not_overwritten(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(launch, "preflight") as preflight,
        ):
            self.assertEqual(launch.main(["--model", "/missing", "--output-dir", d]), 1)
            preflight.assert_not_called()


if __name__ == "__main__":
    unittest.main()
