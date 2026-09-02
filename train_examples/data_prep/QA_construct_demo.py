import json
from string import Template

# 1. Define the raw data samples you provided
raw_data_strings = [
    """{"index":72,"reactants":[{"name":"Cc1cc(C(F)(F)F)ccc1N","volume":"60 g, 342.86 mol, 1 equiv"}],"products":[{"name":"Cc1cc(C(F)(F)F)cc(Br)c1N","volume":"49 g, 67.57%"}],"reagents":[{"name":"","volume":"60.7 g, 342.86 mmol, 1 equiv"}],"solvents":[{"name":"","volume":"1200 mL"}],"additives":[],"yield":"67.57%","nmr":"","file_name":"US20190270754","reaction_type":["Bromination"],"is_multi_stage":false,"smiles_valid":true,"yield_type":"percent","k":"Step 2: Synthesis of 2-bromo-6-methyl-4-(trifluoromethyl)aniline ","v":" To a stirred solution of 2-methyl-4-(trifluoromethyl)aniline (60 g, 342.86 mol, 1 equiv) in DCM (1200 mL) was added NBS (60.7 g, 342.86 mmol, 1 equiv) in portions at 0\u00b0C. under nitrogen atmosphere. The resulting mixture was stirred for 2 h at 0\u00b0C. under nitrogen atmosphere. The reaction mixture was quenched with water at room temperature. The aqueous layer was extracted with DCM and the combined organic layers were concentrated under reduced pressure. The residue was purified by silica gel column chromatography with PE\/EtOAc (30\/1) as eluent to afford the title compound (49 g, 67.57%) as yellow oil. LC-MS: (ES, m\/z): [M+H]\u207a 254. ","clean_response":[{"reactants":[{"name":"2-methyl-4-(trifluoromethyl)aniline","volume":"60 g, 342.86 mol, 1 equiv"}],"products":[{"name":"2-bromo-6-methyl-4-(trifluoromethyl)aniline","volume":"49 g, 67.57%"}],"reagents":[{"name":"NBS","volume":"60.7 g, 342.86 mmol, 1 equiv"}],"solvents":[{"name":"DCM","volume":"1200 mL"}],"additives":[],"reaction_type":["Bromination"],"yield":"67.57%","is_multi_stage":false,"nmr":""}],"num_product":1,"num_reactants":1,"compact":14,"valid":2,"s_products":"Cc1cc(C(F)(F)F)cc(Br)c1N","s_reactants":"Cc1cc(C(F)(F)F)ccc1N","s_solvents":"","s_reagents":""}""",
    """{"index":76,"reactants":[{"name":"COc1ccc(-c2nc3cc(OC)cc(Br)c3o2)cc1","volume":"300 mg, 0.90 mmol"}],"products":[{"name":"COc1ccc(-c2nc3cc(OC)cc(-c4ccco4)c3o2)cc1","volume":""}],"reagents":[{"name":"","volume":"71 mg, 0.09 mmol"}],"solvents":[{"name":"Cc1ccc(C)cc1","volume":"3 mL"}],"additives":[{"name":"CCCC[Sn](CCCC)(CCCC)c1ccco1","volume":"449 mg, 1.26 mmol"}],"yield":"99%","nmr":"","file_name":"US20060046968","reaction_type":["Stille Coupling"],"is_multi_stage":false,"smiles_valid":true,"yield_type":"percent","k":"Step a) 7-(2-Furyl)-5-methoxy-2-(4-methoxyphenyl)-1,3-benzoxazole ","v":" 7-Bromo-5-methoxy-2-(4-methoxyphenyl)-1,3-benzoxazole (300 mg, 0.90 mmol) and dichlorobis(tri-o-tolylphosphine)palladium(II) (71 mg, 0.09 mmol) were dissolved in p-xylene (3 mL) and stirred for 10 mins. at room temperature under a nitrogen atmosphere. 2-(Tributylstannyl)furan (449 mg, 1.26 mmol) was added and the reaction mixture was refluxed for 4 hours. The reaction mixture was cooled to room temperature, diluted with a saturated solution of ammonium chloride and extracted with EtOAc. The organic extracts were washed with water, then brine and dried over MgSO\u2084 and concentrated. Purification by flash chromatography (20%-30% EtOAc\/petroleum ether) gave the title compound as a white solid (99% yield, m.p. 120-121\u00b0C.); MS m\/e 322 (M+H)\u207a.   Analysis for: C\u2081\u2089H\u2081\u2085NO\u2084 Calcd: C, 71.02; H, 4.71; N, 4.36 Found: C, 70.23; H, 4.7; N, 4.19 ","clean_response":[{"reactants":[{"name":"7-Bromo-5-methoxy-2-(4-methoxyphenyl)-1,3-benzoxazole","volume":"300 mg, 0.90 mmol"}],"products":[{"name":"7-(2-Furyl)-5-methoxy-2-(4-methoxyphenyl)-1,3-benzoxazole","volume":""}],"reagents":[{"name":"dichlorobis(tri-o-tolylphosphine)palladium(II)","volume":"71 mg, 0.09 mmol"}],"solvents":[{"name":"p-xylene","volume":"3 mL"}],"additives":[{"name":"2-(Tributylstannyl)furan","volume":"449 mg, 1.26 mmol"}],"reaction_type":["Stille Coupling"],"yield":"99%","is_multi_stage":false,"nmr":""}],"num_product":1,"num_reactants":1,"compact":14,"valid":2,"s_products":"COc1ccc(-c2nc3cc(OC)cc(-c4ccco4)c3o2)cc1","s_reactants":"COc1ccc(-c2nc3cc(OC)cc(Br)c3o2)cc1","s_solvents":"Cc1ccc(C)cc1","s_reagents":""}"""
]

# 2. Define the exact templates based on your target output examples
templates = {
    # Templates for sample_idx == 3 (Only Reactants and Products are valid)
    "retro_idx3": Template("Act as Chemical synthesis specialist. Reverse the arrow of synthesis: product SMILES \u2192 reactant SMILES Output: SMILES only. Formulate precursors for target molecules with SMILES: $Products using bond disconnections. Propose reactants:"),
    "forward_idx3": Template("Act as Chemical synthetic design scientist. Forward the arrow of synthesis: reactant SMILES \u2192 product SMILES Output the SMILES of the required reactants. Considering the reactants $Reactants, what product(s) is formed?"),
    
    # Templates for sample_idx == 1 (Reactants, Products, and Solvents are valid)
    "retro_idx1": Template("Mode: Specialist in organic chemistry. === Retrosynthetic Task === Output: SMILES only. Construct building blocks for target compound(s) with SMILES: $Products optimizing for yield. Suggest starting materials:"),
    "forward_idx1": Template("Organic chemist. Run synthetic movie forward. Respond with SMILES strings. When we have reactants like $Reactants and solvents $Solvents, what product(s) this reaction yields"),
    "condition_idx1": Template("Mode: Chemical synthesis specialist. === Organic Generation Condition Prediction Task === Speak only SMILES. The organic syntheis involves reactant SMILES $Reactants and product SMILES $Products. What solvents are demanded?")
}

# 3. Helper functions from your pipeline
def valid_SMILES(element):
    return len(element) > 0 if element is not None else False

def classify(row):
    s_r, s_p = valid_SMILES(row.get("s_reactants")), valid_SMILES(row.get("s_products"))
    s_s, s_reagents = valid_SMILES(row.get("s_solvents")), valid_SMILES(row.get("s_reagents"))
    if s_r and s_p and s_s and s_reagents: return 0
    elif s_r and s_p and s_s: return 1
    elif s_r and s_p and s_reagents: return 2
    elif s_r and s_p: return 3
    else: return -1

# 4. Main processing loop
def generate_qa_pairs():
    final_outputs = []
    
    for row_str in raw_data_strings:
        row = json.loads(row_str)
        sample_idx = classify(row)
        idx = row["index"]
        
        if sample_idx == 3:
            # Sample 72 falls here (s_solvents and s_reagents are empty strings)
            retro_instruction = templates["retro_idx3"].safe_substitute(Products=row["s_products"])
            final_outputs.append({"sample_idx": sample_idx, "Task type": "Retro", "idx": idx, "instruction": retro_instruction, "output": row["s_reactants"]})
            
            forward_instruction = templates["forward_idx3"].safe_substitute(Reactants=row["s_reactants"])
            final_outputs.append({"sample_idx": sample_idx, "Task type": "Forward", "idx": idx, "instruction": forward_instruction, "output": row["s_products"]})

        elif sample_idx == 1:
            # Sample 76 falls here (s_solvents is populated, s_reagents is empty)
            retro_instruction = templates["retro_idx1"].safe_substitute(Products=row["s_products"])
            final_outputs.append({"sample_idx": sample_idx, "Task type": "Retro", "idx": idx, "instruction": retro_instruction, "output": row["s_reactants"]})
            
            forward_instruction = templates["forward_idx1"].safe_substitute(Reactants=row["s_reactants"], Solvents=row["s_solvents"])
            final_outputs.append({"sample_idx": sample_idx, "Task type": "Forward", "idx": idx, "instruction": forward_instruction, "output": row["s_products"]})
            
            condition_instruction = templates["condition_idx1"].safe_substitute(Reactants=row["s_reactants"], Products=row["s_products"])
            final_outputs.append({"sample_idx": sample_idx, "Task type": "Condition", "idx": idx, "instruction": condition_instruction, "output": f"solvents: {row['s_solvents']}"})

    return final_outputs

if __name__ == "__main__":
    results = generate_qa_pairs()
    
    print("=== FINAL JSONL OUTPUT ===\n")
    for res in results:
        print(json.dumps(res))
