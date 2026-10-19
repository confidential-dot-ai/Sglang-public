import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/ccd_debug_ticks_task133.jsonl"
recs = [json.loads(l) for l in open(path)]
print(f"total records: {len(recs)}")

by_tick = defaultdict(list)
for r in recs:
    by_tick[(r["tag"], r["tick"])].append(r)

for (tag, tick), group in sorted(by_tick.items(), key=lambda kv: kv[0][1]):
    cls = set(tuple(g["commit_lens"]) for g in group)
    toks = set(tuple(g["out_tokens"]) for g in group)
    consistent = len(cls) == 1 and len(toks) == 1
    cl = group[0]["commit_lens"][0] if group[0]["commit_lens"] else 0
    committed = group[0]["out_tokens"][:cl] if cl else []
    flag = "" if consistent else "  <-- RANKS DISAGREE"
    print(f"tick={tick:4d} tag={tag:10s} n_ranks={len(group):2d} commit_len={cl:3d} committed={committed}{flag}")
