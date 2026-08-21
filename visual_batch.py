import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

from rdkit.Chem import AllChem, Draw


def visualize_protocol_reaction(protocol_data, output_filepath, product_str=None):
    """
    Parses a single protocol dictionary, constructs a reaction SMILES,
    and renders a high-quality visualization using RDKit.
    """
    try:
        reaction_conds = protocol_data.get("reaction_conditions", {})

        reactants = reaction_conds.get("reactants", [])
        if not reactants:
            print(f"Skipping step: No reactants found for {output_filepath}")
            return False
        reactant_smiles = [r["smiles"] for r in reactants if r.get("smiles")]
        reactants_str = ".".join(reactant_smiles)

        agent_smiles = []
        if "reagents" in reaction_conds:
            agent_smiles.extend(
                [r["smiles"] for r in reaction_conds["reagents"] if r.get("smiles")]
            )
        if "solvents" in reaction_conds:
            agent_smiles.extend(
                [s["smiles"] for s in reaction_conds["solvents"] if s.get("smiles")]
            )
        agents_str = ".".join(agent_smiles)

        if not product_str:
            print(f"Skipping step: Target molecule SMILES not found for {output_filepath}")
            return False

        reaction_smiles = f"{reactants_str}>{agents_str}>{product_str}"

        rxn = AllChem.ReactionFromSmarts(reaction_smiles, useSmiles=True)
        if rxn is None:
            print(f"RDKit parsing failed for reaction SMILES: {reaction_smiles}")
            return False

        img = Draw.ReactionToImage(rxn, subImgSize=(300, 300), useSVG=False)
        img.save(output_filepath)
        return True

    except Exception as e:
        print(f"Error rendering recipe step for {output_filepath}: {e}")
        return False


def get_protocols(data: dict) -> Optional[List[dict]]:
    """Support both OSCAR_main_port (full_recipe) and MCP client (final_protocol)."""
    if not isinstance(data, dict):
        return None
    protocols = data.get("full_recipe") or data.get("final_protocol")
    if isinstance(protocols, list) and protocols:
        return protocols
    return None


def get_product_smiles(data: dict, fallback: Optional[str] = None) -> Optional[str]:
    return data.get("patent_products") or data.get("products_SMILES") or fallback


def batch_process_json_directory(
    input_dir,
    output_dir,
    product_str=None,
    pattern: str = "*.json",
):
    """
    Scan a directory for JSON files and render each protocol step to PNG.

    Supports:
    - full_recipe (OSCAR_main_port / run_app output)
    - final_protocol (OSCAR_mcp_client output)

    Each JSON file gets its own subfolder under output_dir:
      output_dir/client10.1/step_1.png, step_2.png, ...
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_path.glob(pattern))
    print(f"Input:  {input_path.resolve()}")
    print(f"Output: {output_path.resolve()}")
    print(f"Pattern: {pattern} -> {len(json_files)} file(s)\n")

    total_files = 0
    total_images = 0

    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Skip (invalid JSON): {json_file.name}")
            continue

        protocols = get_protocols(data)
        if not protocols:
            print(f"Skip (no full_recipe / final_protocol): {json_file.name}")
            continue

        product = get_product_smiles(data, product_str)
        if not product:
            print(f"Skip (no product SMILES): {json_file.name}")
            continue

        file_out_dir = output_path / json_file.stem
        file_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {json_file.name} -> {file_out_dir.name}/ ({len(protocols)} step(s))")

        ok_count = 0
        for idx, protocol in enumerate(protocols):
            out_png = file_out_dir / f"step_{idx + 1}.png"
            if visualize_protocol_reaction(protocol, str(out_png), product_str=product):
                ok_count += 1

        total_files += 1
        total_images += ok_count
        print(f"  Done: {ok_count}/{len(protocols)} image(s)\n")

    print(
        f"Batch complete: {total_files} JSON file(s), "
        f"{total_images} image(s) -> {output_path.resolve()}"
    )
    return total_files, total_images


def batch_process_all_molecules(
    output_root: str | Path,
    json_subdir: str = "vllm",
    rendered_subdir: str = "rendered",
    pattern: str = "*.json",
):
    """Process every molecule folder under output_root/*/json_subdir/."""
    root = Path(output_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Output root not found: {root.resolve()}")

    grand_total_files = 0
    grand_total_images = 0

    for mol_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        input_dir = mol_dir / json_subdir
        if not input_dir.is_dir():
            continue

        output_dir = mol_dir / rendered_subdir
        product = mol_dir.name
        print(f"\n{'=' * 60}")
        print(f"Molecule: {product}")
        print(f"{'=' * 60}")

        n_files, n_images = batch_process_json_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            product_str=product,
            pattern=pattern,
        )
        grand_total_files += n_files
        grand_total_images += n_images

    print(
        f"\nAll molecules done: {grand_total_files} JSON file(s), "
        f"{grand_total_images} image(s) under {root.resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch render reaction diagrams from protocol JSON files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing JSON files (e.g. output/<SMILES>/vllm)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Root directory for rendered images (each JSON -> subfolder)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        help="Process all molecule folders under output/ (uses */vllm -> */rendered)",
    )
    parser.add_argument(
        "--product",
        type=str,
        default=None,
        help="Product SMILES fallback when not present in JSON",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help='Glob pattern, e.g. "*.json", "client*.json", "port*.json"',
    )
    parser.add_argument(
        "--json_subdir",
        type=str,
        default="vllm",
        help="JSON subfolder name under each molecule dir (with --output_root)",
    )
    parser.add_argument(
        "--rendered_subdir",
        type=str,
        default="rendered",
        help="Rendered output subfolder name (with --output_root)",
    )
    args = parser.parse_args()

    if args.output_root:
        batch_process_all_molecules(
            output_root=args.output_root,
            json_subdir=args.json_subdir,
            rendered_subdir=args.rendered_subdir,
            pattern=args.pattern,
        )
    elif args.input_dir and args.output_dir:
        batch_process_json_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            product_str=args.product,
            pattern=args.pattern,
        )
    else:
        parser.error("Provide either --output_root OR both --input_dir and --output_dir")
