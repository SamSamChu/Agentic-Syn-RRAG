from mcp.server.fastmcp import FastMCP
import sys
import logging
import pandas as pd
import json
import requests
from contextlib import redirect_stdout
from save_load_embedding import load_retrieval_system
from database_embedding import retrieve_similar_reactions
from self_refine_loop_agent import self_refine_loop

# 1. Force global redirection immediately
_original_stdout = sys.stdout
sys.stdout = sys.stderr

##Log system initialization
import logging
##Basic log configuration: INFO level, includes timestamp, module name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


mcp = FastMCP("Chemistry_RAG")

# Load your pre-built FAISS index
# --- LOAD YOUR SYSTEM ONCE AT STARTUP ---
#logger.info("Loading index and map from local files...")
logger.info("Loading index and map from local files...")
# Assuming these functions are imported from your existing script
with redirect_stdout(sys.stderr):
    faiss_index, indices_map, df_loaded = load_retrieval_system()
logger.info("Done, load the json file")

# CACTUS API 地址
CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"

def smiles_to_iupac(smiles: str) -> str:
    rep = "iupac_name"
    url = CACTUS.format(smiles, rep)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text.strip()


@mcp.tool()
def smiles_to_iupac_tool(smiles: str) -> str:
    try:
        # validation = validate_smiles(smiles)
        # if not validation.get("valid", False):
        #     return f"Error: Invalid SMILES - {validation.get('error')}"
        
        return smiles_to_iupac(smiles)
    except Exception as e:
        return f"Error: {str(e)}"

import json

@mcp.tool()
async def search_similar_reactions(reactants_smiles: str, products_smiles: str, k=10) -> str:
    """Retrieves top 5 similar reactions for relevant documents of organic synthesis experiments description in the knowledge base."""
    # Retrieve top 5 relevant documents
    with redirect_stdout(sys.stderr):
        results = await retrieve_similar_reactions(
            reactants_smiles, 
            products_smiles, 
            faiss_index, 
            indices_map, 
            df_loaded, 
            k=k
        )
    
    # Build query string
    query = f"{reactants_smiles}>>{products_smiles}"
    
    
    # Convert results to list of dicts if it's a DataFrame
    if isinstance(results, pd.DataFrame):
        results_list = results.to_dict('records')
    elif isinstance(results, list):
        results_list = results
    else:
        # Fallback: convert to DataFrame first, then to list
        df_results = pd.DataFrame(results)
        results_list = df_results.to_dict('records')
    
    # Build the JSON structure matching top10_l2.json format
    data = {
        'query': query,
        'results': [
            {
                'reactants': r.get('reactants', ''),
                'products': r.get('products', ''),
                'solvents':r.get('solvents', ''),
                'reagents':r.get('reagents',''),
                'experiments_details': r.get('experiments_details', ''),
                'file_name': r.get('file_name', '')
            }
            for r in results_list
        ]
    }
    
    # Convert to JSON string - 使用 ensure_ascii=False 保留非ASCII字符，但不使用 indent 避免换行问题
    json_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    return json_str

@mcp.tool()
async def execute_synthesis_refinement(product_smiles: str, num_samples: int) -> dict:
    """
    Exposes the self-refining synthesis loop as a tool.
    Accepts a molecule's SMILES string and returns refined pathways include reactants, reagents, solvents, verified also.
    """
    # 1. Compile your existing graph inside the tool or globally
    app = self_refine_loop()

    # 2. Prepare the initial state matching your SynthesisState
    inputs = {
        "messages": [("user", "Refine this reaction")],
        "product_smiles": product_smiles,
        "num_samples": num_samples,
        "verified_results": []
    }

    # 3. Execute the graph (the nodes will still call their internal tools)
    final_state = await app.ainvoke(inputs)

    # 4. Return the aggregated results back to the MCP client
    return {"refined_pathways": final_state.get("verified_results", [])}

if __name__ == "__main__":
    # Use mcp.run() to start the stdio server
    sys.stdout = _original_stdout
    mcp.run()

