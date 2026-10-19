"""Compare graph-off/on continuations in the same loaded CED process."""

import argparse
import importlib.util
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8000, 20000, 40000])
    parser.add_argument("--decode-url", default="http://127.0.0.1:8042")
    parser.add_argument(
        "--component", choices=("all", "encoder", "late"), default="all"
    )
    parser.add_argument("--repeat-eager", action="store_true")
    parser.add_argument("--modes", type=int, choices=[0, 1], nargs="+", default=[0, 1])
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "compare", "/data/dsv41-bringup/lcgraph/compare.py"
    )
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    prompts = dict(compare.prompts())

    def graph(enabled):
        if args.repeat_eager:
            enabled = False
        options = {}
        if args.component in ("all", "encoder"):
            options["encoder_graph"] = enabled
        if args.component in ("all", "late"):
            options["late_graph"] = enabled
        result = compare.post(
            args.decode_url + "/set_internal_state",
            dict(server_args=dict(ced_tail_options=options)),
        )
        assert result and all(value is True for value in result), result

    rows = []
    try:
        for size in args.sizes:
            results = []
            for run, mode in enumerate(args.modes):
                enabled = bool(mode) and not args.repeat_eager
                graph(enabled)
                urlopen(Request(compare.PREFILL + "/flush_cache", method="POST")).read()
                result = compare.post(
                    compare.ROUTER + "/generate",
                    dict(
                        input_ids=prompts[size],
                        return_logprob=True,
                        top_logprobs_num=5,
                        sampling_params=dict(
                            temperature=0, max_new_tokens=24, ignore_eos=True
                        ),
                    ),
                )
                (args.output / f"ab-{size}-run-{run}-graph-{enabled}.json").write_text(
                    json.dumps(result)
                )
                results.append(result)
            a = results[0]
            row = dict(
                prompt_tokens=size,
                modes=[0 if args.repeat_eager else m for m in args.modes],
                output_ids_equal=all(
                    a["output_ids"] == b["output_ids"] for b in results
                ),
                output_logprobs_equal=all(
                    a["meta_info"]["output_token_logprobs"]
                    == b["meta_info"]["output_token_logprobs"]
                    for b in results
                ),
                ids_equal_to_first=[
                    a["output_ids"] == b["output_ids"] for b in results
                ],
            )
            rows.append(row)
            print(row, flush=True)
            (args.output / "ab-comparison.json").write_text(json.dumps(rows, indent=2))
    finally:
        graph(True)
    assert all(row["output_ids_equal"] for row in rows), rows


if __name__ == "__main__":
    main()
