#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.reaction_plausibility import check_reactant_scaffold_conservation

RDLogger.DisableLog("rdApp.*")


TOP_KS = (1, 3, 5, 10)
SCAFFOLD_TIMEOUT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate local, agent, and agent+alt reactant predictions from "
            "run_app shard outputs. Reactants and reagents are merged for "
            "reactant metrics, because models sometimes place true reactants "
            "under reagent fields."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing shard_*/<model>.json outputs.",
    )
    parser.add_argument(
        "--pattern",
        default="shard_*/01_gemini25_top10_model.json",
        help="Glob pattern relative to input_dir. Default: shard_*/01_gemini25_top10_model.json",
    )
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Optional prefix for writing *_summary.json and *_per_record.csv.",
    )
    parser.add_argument(
        "--top_ks",
        default="1,3,5,10",
        help="Comma-separated top-k values. Default: 1,3,5,10",
    )
    parser.add_argument(
        "--scaffold_timeout",
        type=int,
        default=5,
        help="Timeout in seconds passed to check_reactant_scaffold_conservation. Default: 5",
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


def raw_smiles_entries(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        entries = []
        for item in value:
            if isinstance(item, dict):
                entries.append(str(item.get("smiles", "") or ""))
            else:
                entries.append(str(item or ""))
        return entries
    return [str(value or "")]


def split_raw_components(value: Any) -> List[str]:
    components: List[str] = []
    for entry in raw_smiles_entries(value):
        for part in re.split(r"\s*[.|]\s*", entry):
            part = part.strip()
            if part:
                components.append(part)
    return components


def merge_smiles_values(*values: Any) -> str:
    components: List[str] = []
    for value in values:
        components.extend(split_raw_components(value))
    return ".".join(components)


@lru_cache(maxsize=None)
def canonicalize_component(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(str(smiles or "").strip())
    except Exception:
        return None
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


@lru_cache(maxsize=None)
def canonical_components_from_parts(parts: Tuple[str, ...]) -> Optional[Tuple[str, ...]]:
    if not parts:
        return tuple()
    canonical = []
    for part in parts:
        item = canonicalize_component(part)
        if item is None:
            return None
        canonical.append(item)
    return tuple(sorted(canonical))


def canonical_components(value: Any) -> Optional[List[str]]:
    raw = tuple(split_raw_components(value))
    result = canonical_components_from_parts(raw)
    if result is None:
        return None
    return list(result)


@lru_cache(maxsize=None)
def canonical_join_from_parts(parts: Tuple[str, ...]) -> Optional[str]:
    items = canonical_components_from_parts(parts)
    if items is None:
        return None
    return ".".join(items)


def canonical_join(value: Any) -> Optional[str]:
    return canonical_join_from_parts(tuple(split_raw_components(value)))


@lru_cache(maxsize=None)
def elements_from_parts(parts: Tuple[str, ...]) -> Optional[frozenset[str]]:
    if not parts:
        return frozenset()
    found = set()
    for smiles in parts:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        found.update(atom.GetSymbol() for atom in mol.GetAtoms())
    return frozenset(found)


def elements(smiles_values: Any) -> Optional[set[str]]:
    result = elements_from_parts(tuple(split_raw_components(smiles_values)))
    if result is None:
        return None
    return set(result)


@lru_cache(maxsize=None)
def element_consistent_from_parts(reactant_parts: Tuple[str, ...], product: str) -> bool:
    reactant_elements = elements_from_parts(reactant_parts)
    product_elements = elements_from_parts(tuple(split_raw_components(product)))
    if reactant_elements is None or product_elements is None:
        return False
    return product_elements.issubset(reactant_elements)


def element_consistent(reactants: Any, product: str) -> bool:
    return element_consistent_from_parts(tuple(split_raw_components(reactants)), product)


@lru_cache(maxsize=None)
def scaffold_valid_from_parts(reactant_parts: Tuple[str, ...], product: str) -> bool:
    if not reactant_parts:
        return False
    try:
        result = check_reactant_scaffold_conservation(
            list(reactant_parts),
            product,
            timeout=SCAFFOLD_TIMEOUT,
        )
    except Exception:
        return False
    return bool(result.get("valid"))


def scaffold_valid(reactants: Any, product: str) -> bool:
    return scaffold_valid_from_parts(tuple(split_raw_components(reactants)), product)


@lru_cache(maxsize=None)
def candidate_metrics_from_parts(
    reactant_parts: Tuple[str, ...],
    gold_reactants: str,
    product: str,
) -> Tuple[bool, bool, bool, bool]:
    pred_join = canonical_join_from_parts(reactant_parts)
    pred_set = set(canonical_components_from_parts(reactant_parts) or [])
    gold_set = set(canonical_components(gold_reactants) or [])
    valid = pred_join is not None
    return (
        bool(valid and gold_set and gold_set.issubset(pred_set)),
        bool(valid),
        bool(valid and element_consistent_from_parts(reactant_parts, product)),
        bool(valid and scaffold_valid_from_parts(reactant_parts, product)),
    )


def candidate_metrics(reactants: Any, gold_reactants: str, product: str) -> Dict[str, bool]:
    accuracy, valid, elemental, scaffold = candidate_metrics_from_parts(
        tuple(split_raw_components(reactants)),
        gold_reactants,
        product,
    )
    return {
        "accuracy": accuracy,
        "valid": valid,
        "elemental_consistent": elemental,
        "scaffold_valid": scaffold,
    }


def candidate_metric(reactants: Any, gold_reactants: str, product: str, metric: str) -> bool:
    parts = tuple(split_raw_components(reactants))
    pred_join = canonical_join_from_parts(parts)
    valid = pred_join is not None
    if metric == "valid":
        return bool(valid)
    if not valid:
        return False
    if metric == "accuracy":
        pred_set = set(canonical_components_from_parts(parts) or [])
        gold_components = canonical_components(gold_reactants)
        if gold_components is None:
            return False
        gold_set = set(gold_components)
        return bool(gold_set and gold_set.issubset(pred_set))
    if metric == "elemental_consistent":
        return element_consistent_from_parts(parts, product)
    if metric == "scaffold_valid":
        return scaffold_valid_from_parts(parts, product)
    raise ValueError(f"Unknown metric: {metric}")


def _unused_old_canonical_components(value: Any) -> Optional[List[str]]:
    raw = split_raw_components(value)
    if not raw:
        return []
    canonical = []
    for part in raw:
        item = canonicalize_component(part)
        if item is None:
            return None
        canonical.append(item)
    return sorted(canonical)


def canonical_set(value: Any) -> Optional[set[str]]:
    items = canonical_components(value)
    if items is None:
        return None
    return set(items)



def parse_condition_field(condition: str, field: str) -> str:
    match = re.search(
        field + r"\s*:\s*(.*?)(?:\s+(?:reagents|solvents|catalysts|temperature|time)\s*:|$)",
        str(condition or ""),
        flags=re.I,
    )
    return match.group(1).strip() if match else ""


def get_agent_recipe_lists(record: dict) -> List[Any]:
    recipes = record.get("full_recipe") or record.get("final_reports") or []
    if isinstance(recipes, dict):
        return [recipes]
    if isinstance(recipes, list):
        return recipes
    return [recipes] if recipes else []


def get_agent_reactant_pools(recipe: Any, *, use_alt: bool, include_reagents: bool) -> List[Any]:
    parsed = recipe_obj(recipe)
    conditions = parsed.get("reaction_conditions", {})
    revision = parsed.get("reactant_revision", {})
    if not isinstance(conditions, dict):
        conditions = {}
    if not isinstance(revision, dict):
        revision = {}

    reagents = conditions.get("reagents", [])
    if include_reagents:
        candidates = [merge_smiles_values(conditions.get("reactants", []), reagents)]
    else:
        candidates = [conditions.get("reactants", [])]
    if use_alt and revision.get("changed") is False and revision.get("alternative_reactants"):
        if include_reagents:
            candidates.append(merge_smiles_values(revision.get("alternative_reactants"), reagents))
        else:
            candidates.append(revision.get("alternative_reactants"))
    return candidates


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


def record_source_candidate_groups(
    record: dict,
    source: str,
    *,
    include_reagents: bool,
) -> List[List[Any]]:
    """Return one candidate group per ranked result.

    For agent+alt, one ranked recipe can contain two reactant choices: the
    emitted reaction_conditions.reactants and, when changed=false, the preserved
    reactant_revision.alternative_reactants. Top-k is applied to ranked recipes,
    not to the flattened reactant choices.

    include_reagents should be true for accuracy/valid/elemental consistency
    metrics, because true reactants are sometimes placed under reagent fields.
    It should be false for scaffold_valid, because bases/catalysts/reagents do
    not need to conserve scaffold in the product.
    """
    if source == "local":
        groups = []
        for pathway in record.get("pathways") or []:
            if include_reagents:
                candidate = merge_smiles_values(
                    pathway.get("reactants", ""),
                    parse_condition_field(pathway.get("condition", ""), "reagents"),
                )
            else:
                candidate = pathway.get("reactants", "")
            groups.append([candidate])
        return groups
    if source == "agent":
        return [
            get_agent_reactant_pools(recipe, use_alt=False, include_reagents=include_reagents)[:1]
            for recipe in get_agent_recipe_lists(record)
        ]
    if source == "agent+alt":
        return [
            get_agent_reactant_pools(recipe, use_alt=True, include_reagents=include_reagents)
            for recipe in get_agent_recipe_lists(record)
        ]
    raise ValueError(f"Unknown source: {source}")


def evaluate_records(records: Sequence[dict], top_ks: Sequence[int]) -> Tuple[dict, List[dict]]:
    sources = ("local", "agent", "agent+alt")
    metric_names = ("accuracy", "valid", "elemental_consistent", "scaffold_valid")
    summary: Dict[str, Dict[str, Dict[int, dict]]] = {
        source: {metric: {} for metric in metric_names} for source in sources
    }
    per_record_rows: List[dict] = []

    for source in sources:
        for top_k in top_ks:
            totals = {metric: 0 for metric in metric_names}
            for record in records:
                product = record.get("patent_products", "")
                gold = record.get("patent_reactants", "")
                row = {
                    "idx": record.get("idx"),
                    "source": source,
                    "top_k": top_k,
                }
                for metric in metric_names:
                    include_reagents = metric != "scaffold_valid"
                    groups = record_source_candidate_groups(
                        record,
                        source,
                        include_reagents=include_reagents,
                    )[:top_k]
                    candidates = [candidate for group in groups for candidate in group]
                    row["candidate_count"] = max(row.get("candidate_count", 0), len(candidates))
                    hit = any(
                        candidate_metric(candidate, gold, product, metric)
                        for candidate in candidates
                    )
                    totals[metric] += int(hit)
                    row[metric] = int(hit)
                per_record_rows.append(row)

            denominator = len(records)
            for metric in metric_names:
                count = totals[metric]
                summary[source][metric][top_k] = {
                    "count": count,
                    "total": denominator,
                    "rate": count / denominator if denominator else 0.0,
                }

    return summary, per_record_rows


def print_shard_report(shard_counts: Dict[str, List[int]], duplicates: Dict[int, list]) -> None:
    print("SHARDS")
    for shard, ids in sorted(shard_counts.items()):
        ids = [idx for idx in ids if isinstance(idx, int)]
        if ids:
            print(f"{shard}: {len(ids)} records, idx {ids[0]}-{ids[-1]}")
        else:
            print(f"{shard}: 0 records")

    duplicate_items = {idx: locs for idx, locs in duplicates.items() if len(locs) > 1}
    print(f"\nDuplicate idx count: {len(duplicate_items)}")
    if duplicate_items:
        print(json.dumps(duplicate_items, ensure_ascii=False, indent=2))


def print_summary(summary: dict, top_ks: Sequence[int]) -> None:
    metric_titles = {
        "accuracy": "reactant SMILES accuracy",
        "valid": "reactant SMILES valid ratio",
        "elemental_consistent": "reactant SMILES elemental consistent ratio",
        "scaffold_valid": "reactant SMILES scaffold valid ratio",
    }
    for metric, title in metric_titles.items():
        print(f"\n--- {title} ---")
        header = ["source"] + [f"top-{k}" for k in top_ks]
        print("\t".join(header))
        for source in ("local", "agent", "agent+alt"):
            row = [source]
            for top_k in top_ks:
                item = summary[source][metric][top_k]
                row.append(f"{item['count']}/{item['total']} ({item['rate'] * 100:.2f}%)")
            print("\t".join(row))


def write_outputs(output_prefix: str, summary: dict, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)
    with open(output_prefix + "_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    with open(output_prefix + "_per_record.csv", "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "idx",
            "source",
            "top_k",
            "candidate_count",
            "accuracy",
            "valid",
            "elemental_consistent",
            "scaffold_valid",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    global SCAFFOLD_TIMEOUT
    SCAFFOLD_TIMEOUT = args.scaffold_timeout
    top_ks = tuple(int(item.strip()) for item in args.top_ks.split(",") if item.strip())
    records, shard_counts, duplicates = load_records(args.input_dir, args.pattern)
    ids = sorted(record.get("idx") for record in records if isinstance(record.get("idx"), int))
    missing = sorted(set(range(1, 151)) - set(ids))

    print_shard_report(shard_counts, duplicates)
    print(f"\nLoaded records: {len(records)}")
    print(f"idx range: {ids[0] if ids else None}-{ids[-1] if ids else None}")
    print(f"Missing idx in 1..150: {missing}")

    summary, per_record_rows = evaluate_records(records, top_ks)
    print_summary(summary, top_ks)

    if args.output_prefix:
        write_outputs(args.output_prefix, summary, per_record_rows)
        print(f"\nWrote {args.output_prefix}_summary.json")
        print(f"Wrote {args.output_prefix}_per_record.csv")


if __name__ == "__main__":
    main()
