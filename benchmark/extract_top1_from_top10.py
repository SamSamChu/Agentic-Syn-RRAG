#!/usr/bin/env python
"""Keep rank-1 from a completed Top-10 shard run.

Merges concatenated-JSON shards by idx, truncates aligned pathway fields to
the first entry, and writes JSON + pickle. This is a prefix of the pooled
Top-10 ranking, not an independent --num_samples 1 run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from typing import Any, Dict, List


RANK1_ALWAYS = ("full_recipe", "pathways")
RANK1_IF_LONGER = (
    "final_reports",
    "pathway_id",
    "top5_retrieval",
    "retrieval_first",
    "top3_rerank_extract",
    "top3_rerank_retrieval",
    "history",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Top-10 shard outputs by idx and keep rank 1. "
            "Does not recreate a separate --num_samples 1 run."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Completed run directory containing shard_*/<model>.json.",
    )
    parser.add_argument(
        "--pattern",
        default="shard_*/01_gemini25_top10_model.json",
        help="Glob relative to input_dir.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: <input_dir>/derived/top1_from_top10",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Keep the first K ranked items. Default: 1",
    )
    return parser.parse_args()


def parse_concat_json(path: str) -> List[dict]:
    text = open(path, encoding="utf-8").read()
    decoder, pos, records = json.JSONDecoder(), 0, []
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        obj, pos = decoder.raw_decode(text, pos)
        records.append(obj)
    return records


def keep_rank(record: dict, top_k: int) -> dict:
    for key in RANK1_ALWAYS:
        value = record.get(key)
        if isinstance(value, list):
            record[key] = value[:top_k]
    for key in RANK1_IF_LONGER:
        value = record.get(key)
        if isinstance(value, list) and len(value) > top_k:
            record[key] = value[:top_k]
    return record


def load_by_idx(input_dir: str, pattern: str, top_k: int) -> List[dict]:
    by_idx: Dict[int, dict] = {}
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    for path in files:
        for record in parse_concat_json(path):
            idx = record.get("idx")
            if not isinstance(idx, int):
                continue
            by_idx[idx] = keep_rank(record, top_k)
    return [by_idx[idx] for idx in sorted(by_idx)]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or os.path.join(
        args.input_dir, "derived", "top1_from_top10"
    )
    os.makedirs(output_dir, exist_ok=True)

    records = load_by_idx(args.input_dir, args.pattern, args.top_k)
    if not records:
        raise RuntimeError(
            f"No records found under {args.input_dir} with pattern {args.pattern}"
        )

    print(
        "records:",
        len(records),
        "idx:",
        records[0]["idx"],
        "-",
        records[-1]["idx"],
    )

    out_json = os.path.join(output_dir, "top1_from_top10.json")
    out_pkl = os.path.join(output_dir, "top1_from_top10.pkl")
    with open(out_json, "w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False, indent=4)
            handle.write("\n")
    with open(out_pkl, "wb") as handle:
        for record in records:
            pickle.dump(record, handle)

    print("wrote", out_json, out_pkl)


if __name__ == "__main__":
    main()