from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import rdFMCS


def get_elements_from_smiles(smiles: str) -> set[str]:
    """Return the element symbols present in a valid SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return {atom.GetSymbol() for atom in mol.GetAtoms()}


def is_reaction_element_consistent(
    reactant_smiles_list: Sequence[str],
    product_smiles: str,
) -> bool:
    """Check that every product element occurs in a reactant or reagent."""
    reactant_elements: set[str] = set()
    for smiles in reactant_smiles_list:
        reactant_elements.update(get_elements_from_smiles(smiles))
    product_elements = get_elements_from_smiles(product_smiles)
    return product_elements.issubset(reactant_elements)


@dataclass
class ReactantScaffoldCheck:
    reactant_smiles: str
    passed: bool
    reason: str
    heavy_atoms: int = 0
    mcs_atoms: int = 0
    mcs_bonds: int = 0
    coverage: float = 0.0
    product_coverage: float = 0.0
    unmatched_atoms: Optional[List[Dict[str, Any]]] = None
    mcs_smarts: str = ""


def _canonical_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def _organic_heavy_atom_count(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def _carbon_count(mol: Chem.Mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)


def _fragment_mols(smiles: str) -> List[Chem.Mol]:
    mol = _canonical_mol(smiles)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return list(frags) if frags else [mol]


def _is_small_or_inorganic(mol: Chem.Mol, min_heavy_atoms: int) -> bool:
    return _organic_heavy_atom_count(mol) < min_heavy_atoms or _carbon_count(mol) == 0


def _is_likely_salt_or_counterion(mol: Chem.Mol) -> bool:
    """Skip small organic acid/salt fragments that do not define product scaffold."""
    heavy_atoms = _organic_heavy_atom_count(mol)
    carbon_atoms = _carbon_count(mol)
    if heavy_atoms <= 8 and carbon_atoms <= 2:
        hetero_or_halogen = sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() not in {1, 6}
        )
        if hetero_or_halogen >= 3:
            return True
    return False


def _is_reactive_carbon(mol: Chem.Mol, atom_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetAtomicNum() != 6:
        return False

    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetAtomicNum() in {7, 8, 15, 16} and bond.GetBondType() in {
            Chem.BondType.DOUBLE,
            Chem.BondType.AROMATIC,
        }:
            return True
    return False


def _mapped_anchor_has_external_product_bond(
    reactant: Chem.Mol,
    product: Chem.Mol,
    atom_idx: int,
    mapping: Dict[int, int],
    product_match_atoms: set[int],
) -> bool:
    for neighbor in reactant.GetAtomWithIdx(atom_idx).GetNeighbors():
        nbr_idx = neighbor.GetIdx()
        if nbr_idx not in mapping:
            continue
        mapped_anchor = mapping[nbr_idx]
        product_atom = product.GetAtomWithIdx(mapped_anchor)
        if any(n.GetIdx() not in product_match_atoms for n in product_atom.GetNeighbors()):
            return True
    return False


def _unmatched_atom_violation(
    reactant: Chem.Mol,
    product: Chem.Mol,
    atom_idx: int,
    mapping: Dict[int, int],
    product_match_atoms: set[int],
) -> bool:
    atom = reactant.GetAtomWithIdx(atom_idx)
    if atom.GetAtomicNum() != 6:
        return False

    if not _is_reactive_carbon(reactant, atom_idx):
        return True

    return not _mapped_anchor_has_external_product_bond(
        reactant, product, atom_idx, mapping, product_match_atoms
    )


def _unmatched_components(
    mol: Chem.Mol,
    unmatched_atoms: Sequence[int],
) -> List[set[int]]:
    unmatched = set(unmatched_atoms)
    components: List[set[int]] = []
    seen: set[int] = set()

    for atom_idx in unmatched:
        if atom_idx in seen:
            continue
        stack = [atom_idx]
        component: set[int] = set()
        seen.add(atom_idx)
        while stack:
            current = stack.pop()
            component.add(current)
            for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
                nbr_idx = neighbor.GetIdx()
                if nbr_idx in unmatched and nbr_idx not in seen:
                    seen.add(nbr_idx)
                    stack.append(nbr_idx)
        components.append(component)

    return components


def _has_sulfone_like_atom(mol: Chem.Mol, atom_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetAtomicNum() != 16:
        return False

    oxygen_bonds = 0
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetAtomicNum() == 8:
            oxygen_bonds += 1
    return oxygen_bonds >= 2


def _has_known_auxiliary_motif(mol: Chem.Mol) -> bool:
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 15 and (atom.GetDegree() >= 3 or atom.GetFormalCharge() != 0):
            return True
        if atom.GetFormalCharge() != 0 and atomic_num in {6, 7, 15, 16}:
            return True
        if _has_sulfone_like_atom(mol, atom.GetIdx()):
            return True
    return False


def _component_has_carbonyl_or_hetero_auxiliary(
    mol: Chem.Mol,
    component: set[int],
) -> bool:
    hetero_atoms = 0
    has_carbonyl = False
    for atom_idx in component:
        atom = mol.GetAtomWithIdx(atom_idx)
        atomic_num = atom.GetAtomicNum()
        if atomic_num not in {1, 6}:
            hetero_atoms += 1
        if atomic_num in {15, 16} or atom.GetFormalCharge() != 0:
            return True
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if (
                atomic_num == 6
                and other.GetAtomicNum() in {7, 8, 16}
                and bond.GetBondType() == Chem.BondType.DOUBLE
            ):
                has_carbonyl = True
    return hetero_atoms > 0 and has_carbonyl


def _component_attach_atoms(
    mol: Chem.Mol,
    component: set[int],
    matched_atoms: set[int],
) -> List[int]:
    attach_atoms: List[int] = []
    for atom_idx in component:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in matched_atoms:
                attach_atoms.append(nbr_idx)
    return attach_atoms


def _component_is_plausible_leaving_group(
    mol: Chem.Mol,
    component: set[int],
    matched_atoms: set[int],
) -> bool:
    carbon_atoms = [
        atom_idx
        for atom_idx in component
        if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 6
    ]
    if not carbon_atoms:
        return True

    if any(_has_sulfone_like_atom(mol, atom_idx) for atom_idx in component):
        return True

    if _component_has_carbonyl_or_hetero_auxiliary(mol, component):
        return True

    attach_atoms = _component_attach_atoms(mol, component, matched_atoms)
    if not attach_atoms:
        return False

    # Alkyl or acyl groups attached through hetero atoms are common protecting
    # groups or cleaved auxiliaries. A plain carbon substituent on an aryl/alkyl
    # scaffold is not accepted here, because that is the positional-isomer case
    # this check is meant to catch.
    return any(
        mol.GetAtomWithIdx(anchor_idx).GetAtomicNum() in {7, 8, 15, 16}
        for anchor_idx in attach_atoms
    )


def _allows_leaving_group_loss(
    reactant: Chem.Mol,
    matched_atoms: set[int],
    unmatched_atoms: Sequence[int],
    product_coverage: float,
    mcs_atoms: int,
) -> bool:
    if not unmatched_atoms or mcs_atoms < 3 or product_coverage < 0.10:
        return False

    components = _unmatched_components(reactant, unmatched_atoms)
    if components and all(
        _component_is_plausible_leaving_group(reactant, component, matched_atoms)
        for component in components
    ):
        return True

    # Reagents such as TosMIC and phosphonium ylides often contribute only a
    # small product fragment while a large sulfone/phosphine auxiliary leaves.
    if _has_known_auxiliary_motif(reactant) and mcs_atoms >= 6:
        return True

    return False


def _best_mcs_scaffold_match(
    reactant: Chem.Mol,
    product: Chem.Mol,
    timeout: int,
) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], str, int, int]:
    best_pair = None
    best_score = None

    mcs_configs = [
        {
            "ringMatchesRingOnly": True,
            "completeRingsOnly": True,
            "bondCompare": rdFMCS.BondCompare.CompareOrderExact,
        },
        # Ring closure/opening changes ring membership. Keep element and bond
        # order matching, but allow acyclic precursor atoms to map into a ring.
        {
            "ringMatchesRingOnly": False,
            "completeRingsOnly": False,
            "bondCompare": rdFMCS.BondCompare.CompareOrderExact,
        },
        # Some product tautomers/aromatic forms differ in local bond order.
        # This is a final fallback and is penalized by config_rank.
        {
            "ringMatchesRingOnly": False,
            "completeRingsOnly": False,
            "bondCompare": rdFMCS.BondCompare.CompareAny,
        },
    ]

    best_result = None
    for config_rank, config in enumerate(mcs_configs):
        result = rdFMCS.FindMCS(
            [reactant, product],
            timeout=timeout,
            matchValences=False,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            **config,
        )
        if result.canceled or not result.smartsString or result.numAtoms == 0:
            continue

        pattern = Chem.MolFromSmarts(result.smartsString)
        if pattern is None:
            continue

        reactant_matches = reactant.GetSubstructMatches(pattern, uniquify=True)
        product_matches = product.GetSubstructMatches(pattern, uniquify=True)
        if not reactant_matches or not product_matches:
            continue

        for r_match in reactant_matches:
            for p_match in product_matches:
                mapping = {int(r_match[i]): int(p_match[i]) for i in range(len(r_match))}
                matched = set(r_match)
                product_matched = set(p_match)
                unmatched = [idx for idx in range(reactant.GetNumAtoms()) if idx not in matched]
                violation_count = sum(
                    1
                    for idx in unmatched
                    if _unmatched_atom_violation(reactant, product, idx, mapping, product_matched)
                )
                unmatched_carbon_count = sum(
                    1 for idx in unmatched if reactant.GetAtomWithIdx(idx).GetAtomicNum() == 6
                )
                # Prefer mappings that preserve inert carbon skeleton and place
                # changed carbonyl/imine carbons at product atoms with new bonds.
                score = (
                    violation_count,
                    unmatched_carbon_count,
                    len(unmatched),
                    config_rank,
                    -result.numAtoms,
                    -result.numBonds,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (r_match, p_match)
                    best_result = result

    if best_pair is None or best_result is None:
        return None, None, "", 0, 0

    return (
        best_pair[0],
        best_pair[1],
        best_result.smartsString,
        best_result.numAtoms,
        best_result.numBonds,
    )


def _has_product_contained_in_larger_reactant(
    coverage: float,
    product_coverage: float,
    unmatched_atoms: Sequence[int],
) -> bool:
    # Deprotection / hydrolysis / salt liberation often means the product is
    # nearly a substructure of a larger starting material. In these cases many
    # reactant atoms can be absent from the product, but the product scaffold is
    # still explained well.
    return (
        coverage >= 0.55
        and product_coverage >= 0.90
        and len(unmatched_atoms) <= 12
    )


def check_reactant_scaffold_conservation(
    reactant_smiles_list: Sequence[str],
    product_smiles: str,
    *,
    min_heavy_atoms: int = 5,
    min_coverage: float = 0.65,
    max_unmatched_atoms: int = 4,
    timeout: int = 5,
) -> Dict[str, Any]:
    """
    Check whether the non-reacting carbon scaffold of each substantial reactant
    can be preserved as a connected MCS in the product.

    This is intentionally stricter than element-count validation. It catches
    positional-isomer mistakes such as using para-methyl benzaldehyde when the
    product requires an ortho-methyl aryl fragment: the wrong isomer can match
    the ring, but preserving both substituents would require moving a carbon.
    """
    product = _canonical_mol(product_smiles)
    product_heavy_atoms = _organic_heavy_atom_count(product)
    checks: List[ReactantScaffoldCheck] = []

    for raw_smiles in reactant_smiles_list:
        if not raw_smiles:
            continue
        for frag in _fragment_mols(raw_smiles):
            frag_smiles = Chem.MolToSmiles(frag, canonical=True)
            heavy_atoms = _organic_heavy_atom_count(frag)

            if _is_small_or_inorganic(frag, min_heavy_atoms=min_heavy_atoms):
                checks.append(
                    ReactantScaffoldCheck(
                        reactant_smiles=frag_smiles,
                        passed=True,
                        reason="Skipped small or inorganic reactant fragment.",
                        heavy_atoms=heavy_atoms,
                    )
                )
                continue

            if _is_likely_salt_or_counterion(frag):
                checks.append(
                    ReactantScaffoldCheck(
                        reactant_smiles=frag_smiles,
                        passed=True,
                        reason="Skipped likely salt/counterion fragment.",
                        heavy_atoms=heavy_atoms,
                    )
                )
                continue

            r_match, p_match, smarts, mcs_atoms, mcs_bonds = _best_mcs_scaffold_match(
                frag, product, timeout=timeout
            )
            if r_match is None:
                checks.append(
                    ReactantScaffoldCheck(
                        reactant_smiles=frag_smiles,
                        passed=False,
                        reason="No product subgraph preserves this reactant scaffold.",
                        heavy_atoms=heavy_atoms,
                        mcs_atoms=mcs_atoms,
                        mcs_bonds=mcs_bonds,
                        mcs_smarts=smarts,
                    )
                )
                continue

            matched_atoms = set(r_match)
            product_matched_atoms = set(p_match or [])
            mapping = {int(r_match[i]): int(p_match[i]) for i in range(len(r_match))}
            unmatched = [idx for idx in range(frag.GetNumAtoms()) if idx not in matched_atoms]
            unmatched_info = [
                {
                    "atom_index": int(idx),
                    "symbol": frag.GetAtomWithIdx(idx).GetSymbol(),
                    "degree": int(frag.GetAtomWithIdx(idx).GetDegree()),
                }
                for idx in unmatched
            ]
            coverage = len(matched_atoms) / heavy_atoms if heavy_atoms else 1.0
            product_coverage = (
                len(product_matched_atoms) / product_heavy_atoms
                if product_heavy_atoms
                else 1.0
            )
            bad_unmatched = [
                info
                for info in unmatched_info
                if _unmatched_atom_violation(
                    frag,
                    product,
                    info["atom_index"],
                    mapping,
                    product_matched_atoms,
                )
            ]

            passed = True
            reason = "Reactant scaffold is conserved in product."
            if _has_product_contained_in_larger_reactant(
                coverage,
                product_coverage,
                unmatched,
            ):
                passed = True
                reason = (
                    "Product scaffold is contained in a larger reactant; "
                    "extra atoms are consistent with deprotection or cleavage."
                )
            elif _allows_leaving_group_loss(
                frag,
                matched_atoms,
                unmatched,
                product_coverage,
                len(matched_atoms),
            ):
                passed = True
                reason = (
                    "A product subgraph is conserved; unmatched atoms are "
                    "consistent with leaving/protecting/auxiliary groups."
                )
            elif coverage < min_coverage:
                passed = False
                reason = f"Only {coverage:.2f} of reactant heavy atoms are conserved."
            elif len(unmatched) > max_unmatched_atoms:
                passed = False
                reason = f"Too many unmatched reactant atoms: {len(unmatched)}."
            elif bad_unmatched:
                passed = False
                reason = (
                    "Unmatched carbon atoms indicate a changed carbon skeleton "
                    "or positional-isomer mismatch."
                )

            checks.append(
                ReactantScaffoldCheck(
                    reactant_smiles=frag_smiles,
                    passed=passed,
                    reason=reason,
                    heavy_atoms=heavy_atoms,
                    mcs_atoms=len(matched_atoms),
                    mcs_bonds=mcs_bonds,
                    coverage=coverage,
                    product_coverage=product_coverage,
                    unmatched_atoms=unmatched_info,
                    mcs_smarts=smarts,
                )
            )

    failed = [asdict(check) for check in checks if not check.passed]
    return {
        "valid": len(failed) == 0,
        "product_smiles": product_smiles,
        "checks": [asdict(check) for check in checks],
        "failed": failed,
        "message": "All substantial reactant scaffolds are conserved."
        if not failed
        else "; ".join(check["reason"] for check in failed),
    }
