"""Measure a CED batching variant on an already loaded replica.

Uses the inherited fixed-window sweep unchanged except for artifact paths and
concurrency. Saves only this trial's server log interval beside its scorecards.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen


def post(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    with urlopen(
        Request(
            url, data=data, method="POST", headers={"Content-Type": "application/json"}
        ),
        timeout=60,
    ) as response:
        return response.read().decode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--live", type=Path, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--max-wait-ms", type=int, required=True)
    p.add_argument("--concurrency", type=int, nargs="+", default=[56])
    p.add_argument("--prefill-url", default="http://127.0.0.1:8041")
    p.add_argument("--decode-url", default="http://127.0.0.1:8042")
    p.add_argument(
        "--template", type=Path, default=Path("/data/dsv41-bringup/ced/pp2-batch8")
    )
    p.add_argument(
        "--encoder-graph", action=argparse.BooleanOptionalAction, default=None
    )
    p.add_argument("--late-graph", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    (a.output / "server").mkdir()
    options = dict(
        batch_size=a.batch_size,
        max_wait_ms=a.max_wait_ms,
        late_graph=a.late_graph,
        graph_check=True,
        graph_check_every=False,
    )
    if a.encoder_graph is not None:
        options["encoder_graph"] = a.encoder_graph
    response = post(
        a.decode_url + "/set_internal_state",
        {"server_args": {"ced_tail_options": options}},
    )
    (a.output / "options.json").write_text(
        json.dumps(dict(options=options, response=response, live=str(a.live)), indent=2)
    )
    print(response, flush=True)
    updated = json.loads(response)
    if (
        not isinstance(updated, list)
        or not updated
        or not all(x is True for x in updated)
    ):
        raise RuntimeError("All-rank runtime control rejected: " + response)
    template = a.template
    script = (
        (template / "run-sweep.sh").read_text().replace(str(template), str(a.output))
    )
    script = script.replace(
        "for concurrency in 56;",
        "for concurrency in " + " ".join(map(str, a.concurrency)) + ";",
    )
    (a.output / "run-sweep.sh").write_text(script)
    (a.output / "qualify.py").write_text(
        (template / "qualify.py").read_text().replace(str(template), str(a.output))
    )
    sources = [a.live / "server" / name for name in ("decode.log", "prefill.log")]
    offsets = {path: path.stat().st_size for path in sources}
    try:
        post(a.prefill_url + "/flush_cache")
        with (a.output / "sweep.log").open("w") as log:
            subprocess.run(
                ["bash", str(a.output / "run-sweep.sh")],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        with (a.output / "qualify-post-sweep.log").open("w") as log:
            subprocess.run(
                [sys.executable, str(a.output / "qualify.py")],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    finally:
        for path in sources:
            with path.open("rb") as stream:
                stream.seek(offsets[path])
                (a.output / "server" / path.name).write_bytes(stream.read())
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("summarize_ced_variant.py")),
            str(a.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
