import argparse
import ast
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Notification
from rdkit import Chem
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from OSCAR_main import create_chemistry_app
from run_app import (
    _extract_recipe_from_report,
    _fallback_recipe_from_local_pathway,
    _has_reaction_conditions,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["-u", "./mcp_tools/mcp_rag.py"],
)


def _is_valid_smiles(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return Chem.MolFromSmiles(str(value).strip()) is not None


def _canonical_smiles(value: Any) -> str:
    mol = Chem.MolFromSmiles(str(value).strip())
    if mol is None:
        raise ValueError(f"Invalid target product SMILES: {value}")
    return Chem.MolToSmiles(mol, canonical=True)


def _first_csv_cell(input_csv: Path) -> str:
    with input_csv.open("r", encoding="utf-8-sig") as f:
        first_line = f.readline().strip()
    return first_line.split(",", 1)[0].strip() if first_line else ""


def load_target_products(input_csv: Path, smiles_column: Optional[str] = None) -> pd.DataFrame:
    first_cell = _first_csv_cell(input_csv)
    has_header = not _is_valid_smiles(first_cell)

    if has_header:
        df = pd.read_csv(input_csv)
    else:
        df = pd.read_csv(input_csv, header=None)

    if smiles_column:
        if smiles_column not in df.columns:
            raise ValueError(f"Column '{smiles_column}' not found in {input_csv}")
        product_col = smiles_column
    elif has_header:
        candidates = [
            "smiles",
            "SMILES",
            "target_smiles",
            "product_smiles",
            "products_SMILES",
            "s_products",
            "molecule",
        ]
        product_col = next((col for col in candidates if col in df.columns), None)
        if product_col is None:
            raise ValueError(
                f"Could not infer product SMILES column. Available columns: {list(df.columns)}"
            )
    else:
        product_col = 0

    records = []
    for row_idx, row in df.iterrows():
        raw_smiles = row[product_col]
        if raw_smiles is None or pd.isna(raw_smiles) or not str(raw_smiles).strip():
            continue
        canonical = _canonical_smiles(raw_smiles)
        record = {
            "input_row": int(row_idx) + 1,
            "target_smiles": str(raw_smiles).strip(),
            "canonical_target_smiles": canonical,
        }
        for col in df.columns:
            key = str(col)
            if key not in record:
                value = row[col]
                record[f"input_{key}"] = "" if pd.isna(value) else value
        records.append(record)

    return pd.DataFrame(records)


def _extract_failed_ids(validation_errors: List[Dict[str, Any]]) -> List[Any]:
    failed_ids = []
    for error in validation_errors:
        if isinstance(error, dict) and error.get("type") != "success":
            failed_ids.append(error.get("pathway_id"))
    return failed_ids


def _load_concatenated_json_objects(path: Path) -> List[Dict[str, Any]]:
    """Read either one-line JSONL or pretty-printed concatenated JSON objects."""
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    decoder = json.JSONDecoder()
    records = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        obj, next_index = decoder.raw_decode(text, index)
        records.append(obj)
        index = next_index
    return records


def build_output_entry(
    *,
    input_record: Dict[str, Any],
    result: Dict[str, Any],
    num_samples: int,
) -> Dict[str, Any]:
    pathways = result.get("pathways", []) or []
    pathway_ids = result.get("pathway_id", []) or []
    final_reports = result.get("final_reports", []) or []
    validation_errors = result.get("validation_errors", []) or []

    final_recipes = []
    for i, report in enumerate(final_reports):
        pid = pathway_ids[i] if i < len(pathway_ids) else i
        recipe = _extract_recipe_from_report(report)
        if not _has_reaction_conditions(recipe):
            local_pathway = (
                pathways[pid]
                if isinstance(pid, int) and 0 <= pid < len(pathways)
                else pathways[i]
                if i < len(pathways)
                else {}
            )
            recipe = _fallback_recipe_from_local_pathway(
                local_pathway,
                input_record["canonical_target_smiles"],
            )
            logger.warning(
                "Agent recipe empty/invalid for input_row=%s pathway_id=%s; using local fallback.",
                input_record["input_row"],
                pid,
            )
        final_recipes.append(recipe)

    while len(final_recipes) < len(pathways):
        i = len(final_recipes)
        final_recipes.append(
            _fallback_recipe_from_local_pathway(
                pathways[i],
                input_record["canonical_target_smiles"],
            )
        )

    return {
        "idx": input_record["input_row"],
        "input_row": input_record["input_row"],
        "target_products": input_record["canonical_target_smiles"],
        "patent_products": input_record["canonical_target_smiles"],
        "input_target_smiles": input_record["target_smiles"],
        "canonical_target_smiles": input_record["canonical_target_smiles"],
        "input_metadata": {
            key: value
            for key, value in input_record.items()
            if key not in {"input_row", "target_smiles", "canonical_target_smiles"}
        },
        "num_samples": num_samples,
        "n_pathways": len(pathways),
        "n_full_recipe": len(final_recipes),
        "pathways": pathways,
        "pathway_id": pathway_ids,
        "retrieval_first": result.get("top5_retrieval", []),
        "top3_rerank_retrieval": result.get("top3_rerank_retrieval", []),
        "top3_rerank_extract": result.get("top3_rerank_extract", []),
        "validation_errors": validation_errors,
        "failed_pathway_ids": _extract_failed_ids(validation_errors),
        "full_recipe": final_recipes,
        "final_reports": final_reports,
        "history": result.get("history", []),
    }


async def predict_targets(
    *,
    input_csv: Path,
    output_jsonl: Path,
    smiles_column: Optional[str],
    num_samples: int,
    simple_rag: bool,
    generate_only: bool,
    scaffold_validation: bool,
    max_concurrency: int,
    checkpoint_file: Path,
) -> None:
    targets = load_target_products(input_csv, smiles_column=smiles_column)
    if targets.empty:
        raise ValueError(f"No valid target product SMILES found in {input_csv}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    start_idx = 0
    if checkpoint_file.exists():
        checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        start_idx = int(checkpoint.get("last_completed_position", 0))
        logger.info("Resuming from checkpoint position %s", start_idx)
    elif output_jsonl.exists():
        start_idx = len(_load_concatenated_json_objects(output_jsonl))
        if start_idx:
            logger.info("Resuming from existing output with %s completed records", start_idx)

    run_config = {
        "configurable": {"thread_id": f"target_mols_{output_jsonl.stem}"},
        "max_concurrency": max_concurrency,
    }

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.send_notification(
                Notification(method="notifications/initialized", params={})
            )
            mcp_tools = await load_mcp_tools(session)
            logger.info("Tools loaded: %s", [tool.name for tool in mcp_tools])
            mcp_retrieve = next(tool for tool in mcp_tools if tool.name == "search_similar_reactions")
            mcp_refiner = next(tool for tool in mcp_tools if tool.name == "execute_synthesis_refinement")
            app = create_chemistry_app(
                mcp_retrieve,
                mcp_refiner,
                num_samples=num_samples,
                simple_rag=simple_rag,
                generate_only=generate_only,
            )

            for position, input_record in enumerate(targets.to_dict("records"), start=1):
                if position <= start_idx:
                    continue

                product_smiles = input_record["canonical_target_smiles"]
                inputs = {
                    "messages": [("user", "What are the common conditions for this reaction?")],
                    "products_SMILES": product_smiles,
                    "num_samples": num_samples,
                    "scaffold_validation": scaffold_validation,
                }

                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(5),
                    wait=wait_exponential(multiplier=1, min=4, max=60),
                    reraise=True,
                ):
                    with attempt:
                        logger.info(
                            "Processing target %s/%s input_row=%s product=%s attempt=%s",
                            position,
                            len(targets),
                            input_record["input_row"],
                            product_smiles,
                            attempt.retry_state.attempt_number,
                        )
                        result = await app.ainvoke(inputs, run_config)

                entry = build_output_entry(
                        input_record=input_record,
                        result=result,
                        num_samples=num_samples,
                )
                with output_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False, indent=4) + "\n")
                checkpoint_file.write_text(
                    json.dumps(
                        {
                            "last_completed_position": position,
                            "updated_at": datetime.now().isoformat(timespec="seconds"),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                logger.info("Saved completed target position %s to %s", position, output_jsonl)

    if checkpoint_file.exists():
        checkpoint_file.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict top-k synthesis plans for a CSV list of target small molecules."
    )
    parser.add_argument(
        "-f",
        "--input_csv",
        default="target_mols/target_mol.csv",
        help="Input CSV. Headerless files are supported; the first column is treated as product SMILES.",
    )
    parser.add_argument(
        "-o",
        "--output_jsonl",
        "--output_file",
        default=None,
        help="Output JSONL path. Default: <input_dir>/<input_stem>_top<num_samples>_predictions.jsonl",
    )
    parser.add_argument(
        "--smiles_column",
        default=None,
        help="Product SMILES column name for headered CSV files.",
    )
    parser.add_argument("--num_samples", type=int, default=10, help="Number of top pathways to output.")
    parser.add_argument("--generate_only", action="store_true", help="Disable retrieval ablation mode.")
    parser.add_argument("--simple_rag", action="store_true", help="Use simple RAG pipeline.")
    parser.add_argument(
        "--scaffold_validation",
        "--scaffold-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable scaffold validation in final recipe generation.",
    )
    parser.add_argument("--max_concurrency", type=int, default=8)
    parser.add_argument("--checkpoint_file", default=None)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if args.output_jsonl:
        output_jsonl = Path(args.output_jsonl)
    else:
        output_jsonl = input_csv.with_name(
            f"{input_csv.stem}_top{args.num_samples}_predictions.jsonl"
        )
    checkpoint_file = (
        Path(args.checkpoint_file)
        if args.checkpoint_file
        else output_jsonl.with_suffix(output_jsonl.suffix + ".checkpoint.json")
    )

    asyncio.run(
        predict_targets(
            input_csv=input_csv,
            output_jsonl=output_jsonl,
            smiles_column=args.smiles_column,
            num_samples=args.num_samples,
            simple_rag=args.simple_rag,
            generate_only=args.generate_only,
            scaffold_validation=args.scaffold_validation,
            max_concurrency=args.max_concurrency,
            checkpoint_file=checkpoint_file,
        )
    )


if __name__ == "__main__":
    main()
