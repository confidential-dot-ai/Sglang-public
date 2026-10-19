"""Capture one-shot CED tail traces using the existing long-probe corpus."""

import argparse
import concurrent.futures
import importlib.util
import json
import time
from pathlib import Path
from urllib.request import urlopen

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
a = p.parse_args()
root = a.output
spec = importlib.util.spec_from_file_location(
    "compare", "/data/dsv41-bringup/lcgraph/compare.py"
)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
for _ in range(1800):
    try:
        if (
            json.load(urlopen(compare.ROUTER + "/readiness", timeout=3))[
                "healthy_workers"
            ]
            == 2
        ):
            break
    except Exception:
        pass
    time.sleep(2)
else:
    raise RuntimeError("Replica not ready")
# Let the launcher finish its PD smoke before arming profiling.
time.sleep(10)
prompts = dict(compare.prompts())


def request(ids, tokens=24):
    return compare.post(
        compare.ROUTER + "/generate",
        dict(
            input_ids=ids,
            sampling_params=dict(temperature=0, max_new_tokens=tokens, ignore_eos=True),
            return_logprob=True,
            top_logprobs_num=5,
        ),
    )


for size in (8000, 40000):
    request(prompts[size])  # warm the eager kernel paths before capture
    (root / "traces/trigger").touch()
    result = request(prompts[size])
    (root / f"profile-{size}-output.json").write_text(json.dumps(result))
    print("Captured", size, flush=True)
# Keep a streaming decode active while the next tail arrives.
with concurrent.futures.ThreadPoolExecutor() as pool:
    running = pool.submit(request, prompts[8000], 256)
    time.sleep(0.5)
    (root / "traces/trigger").touch()
    result = request(prompts[40000])
    (root / "profile-light-output.json").write_text(json.dumps(result))
    running.result()
print("Captured under light load", flush=True)
