"""Exercise CED graph checks for coalesced long-prompt tail batch sizes 1–8."""

import argparse
import concurrent.futures
import importlib.util
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--decode-url", default="http://127.0.0.1:8042")
    parser.add_argument(
        "--encoder-graph", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--late-graph", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--restore-wait-ms", type=int, default=1000)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "compare", "/data/dsv41-bringup/lcgraph/compare.py"
    )
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    ids = dict(compare.prompts())[8000]

    def request(tokens):
        return compare.post(
            compare.ROUTER + "/generate",
            dict(
                input_ids=ids,
                sampling_params=dict(
                    temperature=0, max_new_tokens=tokens, ignore_eos=True
                ),
            ),
        )

    def options(n, wait):
        controls = dict(
            batch_size=n, max_wait_ms=wait, late_graph=args.late_graph, graph_check=True
        )
        if args.encoder_graph is not None:
            controls["encoder_graph"] = args.encoder_graph
        updated = compare.post(
            args.decode_url + "/set_internal_state",
            dict(server_args=dict(ced_tail_options=controls)),
        )
        assert updated and all(value is True for value in updated), updated

    rows = []
    request(24)  # Cache the shared prompt before creating each arrival burst.
    try:
        for n in range(1, 9):
            options(n, 5000)
            with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
                anchor = pool.submit(request, 1024)
                time.sleep(0.5)
                futures = [pool.submit(request, 24) for _ in range(n)]
                results = [future.result() for future in futures]
                anchor.result()
            rows.append(dict(requested_batch_size=n, outputs=results))
            args.output.write_text(json.dumps(rows))
            print("Completed arrival burst", n, flush=True)
    finally:
        options(8, args.restore_wait_ms)


if __name__ == "__main__":
    main()
