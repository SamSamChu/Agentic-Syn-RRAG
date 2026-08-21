#!/usr/bin/env python
"""Extract llm_judge_eval_results.json for LLM-as-a-judge scripts.

Reads run_app shard JSON (same layout as evaluate_reactant_metrics.py) or a
single pkl/json file, then writes one record per recipe for:

- test_API_with_self_correction_gpt.py
  (product_SMILES, patent_ground_truth, local_baseline, agent_recipe)
- test_API_with_faithfulness_score.py
  (product_SMILES, rerank_retrieval, agent_recipe)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand run_app records into llm_judge_eval_results.json for "
            "self-correction and faithfulness judge scripts."
        )
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Directory containing shard_*/<model>.json outputs.",
    )
    parser.add_argument(
        "--pattern",
        default="shard_*/01_gemini25_top10_model.json",
        help="Glob pattern relative to input_dir. Default: shard_*/01_gemini25_top10_model.json",
    )
    parser.add_argument(
        "--input_file",
        default=None,
        help="Optional single pkl / json / concat-json file instead of input_dir.",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Output prefix. Writes {prefix}llm_judge_eval_results.json",
    )
    return parser.parse_args()


def parse_concat_json(path: str) -> List[dict]:
    """Parse files written as multiple JSON objects appended back-to-back."""
    text = open(path, encoding="utf-8").read()
    decoder = json.JSONDecoder()
    pos = 0
    records = []
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        obj, end = decoder.raw_decode(text, pos)
        records.append(obj)
        pos = end
    return records


def recipe_obj(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    text = value.strip()
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if match:
        text = match.group(1)
    else:
        match = re.search(r"(\{.*\})", text, flags=re.S)
        if match:
            text = match.group(1)
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def get_agent_recipe_lists(record: dict) -> List[Any]:
    recipes = record.get("full_recipe") or record.get("final_reports") or []
    if isinstance(recipes, dict):
        return [recipes]
    if isinstance(recipes, list):
        return recipes
    return [recipes] if recipes else []


def normalize_recipe(recipe: Any) -> Any:
    if isinstance(recipe, dict):
        return recipe
    parsed = recipe_obj(recipe)
    return parsed if parsed else recipe


def load_pkl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "rb") as f:
        while True:
            try:
                items.append(pickle.load(f))
            except EOFError:
                break
    return items


def load_single_file(path: str) -> List[dict]:
    lower = path.lower()
    if lower.endswith(".pkl") or lower.endswith(".pickle"):
        return load_pkl(path)
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict):
            return [loaded]
    except json.JSONDecodeError:
        pass
    return parse_concat_json(path)


def load_records(input_dir: str, pattern: str) -> Tuple[List[dict], Dict[str, List[int]], Dict[int, list]]:
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    by_idx: Dict[int, dict] = {}
    shard_counts: Dict[str, List[int]] = {}
    duplicates: Dict[int, list] = defaultdict(list)

    for path in files:
        shard = os.path.basename(os.path.dirname(path))
        records = parse_concat_json(path)
        shard_counts[shard] = [record.get("idx") for record in records]
        for offset, record in enumerate(records):
            idx = record.get("idx")
            duplicates[idx].append((shard, offset))
            if isinstance(idx, int):
                by_idx[idx] = record

    return [by_idx[idx] for idx in sorted(by_idx)], shard_counts, duplicates


def pick_baseline(pathways: Sequence[Any], pathway_ids: Sequence[Any], index: int) -> Any:
    pid = pathway_ids[index] if index < len(pathway_ids) else index
    if isinstance(pid, int) and 0 <= pid < len(pathways):
        return pathways[pid]
    if index < len(pathways):
        return pathways[index]
    return ""


def pick_retrieval(reranks: Sequence[Any], n_recipes: int, index: int) -> Any:
    if n_recipes == 1:
        return reranks if reranks else ""
    if index < len(reranks):
        return reranks[index]
    if len(reranks) == 1:
        return reranks[0]
    return reranks if reranks else ""


def expand_record(record: dict) -> List[dict]:
    recipes = [normalize_recipe(recipe) for recipe in get_agent_recipe_lists(record)]
    pathways = as_list(record.get("pathways"))
    pathway_ids = as_list(record.get("pathway_id"))
    reranks = as_list(record.get("top3_rerank_extract"))
    patent_ground_truth = {
        "patent_reactants": record.get("patent_reactants"),
        "patent_solvents": record.get("patent_solvents"),
        "patent_reagents": record.get("patent_reagents"),
        "patent_procedure": record.get("patent_details"),
    }
    product_smiles = record.get("patent_products")
    idx = record.get("idx")

    if not recipes:
        print(f"warning: idx={idx} has no full_recipe/final_reports, skipped")
        return []

    rows = []
    for i, recipe in enumerate(recipes):
        rows.append(
            {
                "idx": idx,
                "product_SMILES": product_smiles,
                "agent_recipe": recipe,
                "local_baseline": pick_baseline(pathways, pathway_ids, i),
                "rerank_retrieval": pick_retrieval(reranks, len(recipes), i),
                "patent_ground_truth": patent_ground_truth,
            }
        )
    return rows


def print_load_report(
    records: Sequence[dict],
    shard_counts: Optional[Dict[str, List[int]]] = None,
    duplicates: Optional[Dict[int, list]] = None,
) -> None:
    if shard_counts:
        print("SHARDS")
        for shard, ids in sorted(shard_counts.items()):
            ids = [idx for idx in ids if isinstance(idx, int)]
            if ids:
                print(f"{shard}: {len(ids)} records, idx {ids[0]}-{ids[-1]}")
            else:
                print(f"{shard}: 0 records")
        duplicate_items = {
            idx: locs for idx, locs in (duplicates or {}).items() if len(locs) > 1
        }
        print(f"Duplicate idx count: {len(duplicate_items)}")

    ids = [record.get("idx") for record in records if isinstance(record.get("idx"), int)]
    print(f"Loaded records: {len(records)}")
    if ids:
        print(f"idx range: {min(ids)}-{max(ids)}")


def main() -> None:
    args = parse_args()
    if not args.input_dir and not args.input_file:
        raise SystemExit("Provide --input_dir or --input_file")

    shard_counts = None
    duplicates = None
    if args.input_file:
        records = load_single_file(args.input_file)
        by_idx: Dict[int, dict] = {}
        for record in records:
            idx = record.get("idx")
            if isinstance(idx, int):
                by_idx[idx] = record
        records = [by_idx[idx] for idx in sorted(by_idx)] if by_idx else records
    else:
        records, shard_counts, duplicates = load_records(args.input_dir, args.pattern)

    print_load_report(records, shard_counts, duplicates)

    json_data: List[dict] = []
    for record in records:
        json_data.extend(expand_record(record))

    out_path = args.prefix + "llm_judge_eval_results.json"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(json_data, handle, indent=4, ensure_ascii=False)

    print(f"Wrote {len(json_data)} judge rows to {out_path}")
    if json_data:
        print(f"Sample keys: {sorted(json_data[0].keys())}")


if __name__ == "__main__":
    main()
