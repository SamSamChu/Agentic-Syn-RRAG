import os
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

def visualize_protocol_reaction(json_data, output_filepath):
    """
    Parses the protocol JSON schema, constructs a reaction SMILES,
    and renders a high-quality visualization using RDKit.
    """
    try:
        # 1. Extract and join reactants
        reactant_smiles = [r['smiles'] for r in json_data['reaction_conditions']['reactants']]
        reactants_str = ".".join(reactant_smiles)
        
        # 2. Extract and join agents (reagents + solvents)
        agent_smiles = []
        if 'reagents' in json_data['reaction_conditions']:
            agent_smiles.extend([r['smiles'] for r in json_data['reaction_conditions']['reagents'] if r.get('smiles')])
        if 'solvents' in json_data['reaction_conditions']:
            agent_smiles.extend([s['smiles'] for s in json_data['reaction_conditions']['solvents'] if s.get('smiles')])
        agents_str = ".".join(agent_smiles)
        
        # 3. Extract target product (parsing from the final procedure text or target variable)
        # Note: If your JSON has an explicit 'product' field, use that. 
        # Otherwise, we pull the final string from your target molecule variable.
        product_str = json_data.get('target_molecule_smiles', "") 
        if not product_str:
            # Fallback placeholder if target is not explicitly mapped in the JSON root
            # Adjust this key to match where you store your target SMILES
            raise ValueError("Target molecule SMILES not found in JSON data.")

        # 4. Assemble the complete Reaction SMILES
        reaction_smiles = f"{reactants_str}>{agents_str}>{product_str}"
        
        # 5. Initialize RDKit Chemical Reaction Object
        #rxn = AllChem.ReactionFromSmarts(reaction_smiles, useSmarts=False)
        rxn = AllChem.ReactionFromSmarts(reaction_smiles, useSmiles=True)
        
        # 6. Draw and save the image
        # ReactionToImage creates a standard PIL Image layout
        img = Draw.ReactionToImage(rxn, subImgSize=(300, 300), useSVG=False)
        img.save(output_filepath)
        print(f"Successfully visualized: {output_filepath}")
        
    except Exception as e:
        print(f"Error rendering reaction: {e}")

# ==========================================
# Example Batch Processing Execution Loop
# ==========================================
if __name__ == "__main__":
    # Create an output directory for your diagrams
    output_dir = "./reaction_diagrams"
    os.makedirs(output_dir, exist_ok=True)
    
    # Mock data representing your exact JSON structure
    sample_protocol = {
        "target_molecule_smiles": "CC1=CC=CC(C2=NC(C=CC=C3)=C3C(N2)=O)=C1",
        "reaction_conditions": {
            "reactants": [
                {"name": "2-Aminobenzamide", "smiles": "NC(=O)c1ccccc1N"},
                {"name": "m-Tolualdehyde", "smiles": "Cc1cccc(C=O)c1"}
            ],
            "reagents": [
                {"name": "Iodine", "smiles": "II"},
                {"name": "Potassium Carbonate", "smiles": "C(=O)([O-])[O-].[K+].[K+]"}
            ],
            "solvents": [
                {"name": "DMSO", "smiles": "CS(C)=O"}
            ]
        }
    }
    
    # Run a single test
    visualize_protocol_reaction(sample_protocol, os.path.join(output_dir, "molecule_1.png"))
    protocol_path = "output/CC1\=CC\=CC\(C2\=NC\(C\=CC\=C3\)\=C3C\(N2\)\=O\)\=C1/env/"


    # To run your full 150 samples, you would simply loop like this:
    # for i, protocol in enumerate(your_150_protocols_list):
    #     visualize_protocol_reaction(protocol, f"{output_dir}/protocol_{i}.png")
