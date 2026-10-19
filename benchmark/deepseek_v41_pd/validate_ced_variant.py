"""Run the existing long probes and oracle qualification for a CED variant."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
p.add_argument("--sweep", action="store_true")
p.add_argument("--router-url", default="http://127.0.0.1:8040")
p.add_argument("--prefill-url", default="http://127.0.0.1:8041")
p.add_argument(
    "--reference", type=Path, default=Path("/data/dsv41-bringup/ced/pp2/probe1.json")
)
p.add_argument(
    "--allow-20k-drift",
    action="store_true",
    help="Record (never call exact) the known 20K continuation drift; require identical first token/logprob",
)
a = p.parse_args()
root = a.output
for _ in range(360):
    try:
        with urlopen(a.router_url + "/readiness", timeout=3) as r:
            if json.load(r).get("healthy_workers") == 2:
                break
    except Exception:
        pass
    time.sleep(5)
else:
    raise RuntimeError("Replica did not become ready")


def run(name, command):
    with (root / (name + ".log")).open("w") as f:
        subprocess.run(
            command,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            env={
                **os.environ,
                "PROBE_ROUTER": a.router_url,
                "PROBE_PREFILL": a.prefill_url,
            },
        )


run(
    "probe",
    [
        sys.executable,
        "/data/dsv41-bringup/lcgraph/compare.py",
        str(root / "probe.json"),
    ],
)
reference = json.loads(a.reference.read_text())
actual = json.loads((root / "probe.json").read_text())
comparison = [
    dict(
        prompt_tokens=x["prompt_tokens"],
        output_ids_equal=x["output_ids"] == y["output_ids"],
        first_token_equal=x["first"][1] == y["first"][1],
        first_logprob_delta=x["first"][0] - y["first"][0],
    )
    for x, y in zip(actual, reference)
]
(root / "probe-comparison.json").write_text(json.dumps(comparison, indent=2))
print(json.dumps(comparison), flush=True)
assert len(actual) == len(reference) and all(
    x["output_ids_equal"]
    or (
        a.allow_20k_drift
        and x["prompt_tokens"] == 20000
        and x["first_token_equal"]
        and x["first_logprob_delta"] == 0.0
    )
    for x in comparison
)
run("qualify", [sys.executable, str(root / "qualify.py")])
print(
    "Probe checks and exact oracle passed; all probe IDs exact:",
    all(x["output_ids_equal"] for x in comparison),
    flush=True,
)
if a.sweep:
    # Match the inherited benchmark's cold initial prefill cache.
    from urllib.request import Request

    urlopen(Request(a.prefill_url + "/flush_cache", method="POST"), timeout=60).read()
    run("sweep", ["bash", str(root / "run-sweep.sh")])
    run("qualify-post-sweep", [sys.executable, str(root / "qualify.py")])
    print("Sweep and post-load oracle passed", flush=True)
