import pandas as pd
import numpy as np
import faiss
import asyncio
import os
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import os, uuid, tempfile
from contextlib import contextmanager
from rxngraphormer.rxn_emb import RXNEMB
from multiprocessing import Pool

##Log system initialization
import logging
##Basic log configuration: INFO level, includes timestamp, module name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


embedding_lock = asyncio.Lock()
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# pretrain_model_path = "/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/1989-chenxin/OSCAR_agentic/pretrained_classification_model"
pretrain_model_path = os.path.join(_BASE_DIR, 'pretrained_classification_model')
rxnemb = RXNEMB(pretrained_model_path=pretrain_model_path, model_type="classifier")

# Per-process scratch dir so parallel shards do not share ./rxn_emb_tmp
RXN_EMB_ROOT = os.path.join(tempfile.gettempdir(), f"rxn_emb_{uuid.uuid4().hex[:8]}")
os.makedirs(RXN_EMB_ROOT, exist_ok=True)
logger.info("RXN embedding temp root: %s", RXN_EMB_ROOT)


@contextmanager
def _in_rxn_emb_root():
    old = os.getcwd()
    os.chdir(RXN_EMB_ROOT)
    try:
        yield
    finally:
        os.chdir(old)


def prepare_offline_data(df: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:

    embeddings = []
    indices_map = []
    #embeddings, indices_map = run_parallel(df,pretrain_model_path)
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating Embeddings"):  #df.iterrows():
        try:
            reaction_smiles = f"{row['s_reactants']}>>{row['s_products']}"
            print(reaction_smiles)
            with _in_rxn_emb_root():
                reaction_embedding = rxnemb.gen_rxn_emb([reaction_smiles, reaction_smiles])
 
            reaction_embedding = reaction_embedding[0]
            embeddings.append(reaction_embedding)
            indices_map.append(index)
        except Exception as e:
            # Handle molecules RDKit can't process
            logger.warning("Skipping index %s: %s", index, e)
            continue
    return np.array(embeddings), indices_map


# --- 2. Function to build the retrieval index ---

def build_retrieval_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds an efficient FAISS index for the given embeddings.

    Args:
        embeddings: A numpy array of pre-computed reaction embeddings.

    Returns:
        A trained FAISS index object.
    """
    dimension = embeddings.shape[1]
    # Use IndexFlatL2 for L2 (Euclidean) distance search.
    # For cosine similarity, you would normalize vectors and use IndexFlatIP (Inner Product).
    index = faiss.IndexFlatL2(dimension)
    
    logger.info("Adding %d embeddings to FAISS index...", embeddings.shape[0])
    index.add(embeddings)
    logger.info("Index ready.")
    
    return index


# --- 3. Function to perform retrieval for a query ---

async def retrieve_similar_reactions(
    query_reactants_smi: str,
    query_products_smi: str,
    index: faiss.Index,
    indices_map: List[int],
    original_df: pd.DataFrame,
    k: int = 100
) -> List[Dict[str, Any]]:
    """
    Searches the FAISS index for the top-k most similar reactions.

    Args:
        query_reactants_smi: SMILES string for query reactants.
        query_products_smi: SMILES string for query products.
        index: The trained FAISS index.
        indices_map: Mapping list from index ID to original DF index.
        original_df: The DataFrame containing full experiment details.
        k: Number of neighbors to retrieve (top-100).

    Returns:
        A list of dictionaries containing retrieved reaction details and similarity scores.
    """
    # 1. Generate the query embedding in the same way as the training data
    try:
        query_rxn_smiles = f"{query_reactants_smi}>>{query_products_smi}"
        # Use the lock to ensure only one thread/process calls the embedding generator at a time
        async with embedding_lock:
            # This prevents 'persistent_load' and 'file not found' errors
            # caused by overlapping temp file access (also isolates parallel shards)
            with _in_rxn_emb_root():
                embeddings = rxnemb.gen_rxn_emb([query_rxn_smiles, query_rxn_smiles])
        #embeddings = rxnemb.gen_rxn_emb([query_rxn_smiles,query_rxn_smiles])
        query_embedding = embeddings[0:1]
    
    except Exception as e:
        logger.error("Error processing query SMILES: %s", e)
        return []

    # 2. Perform the FAISS search
    # D = distances, I = indices in the FAISS index
    distances, faiss_indices = index.search(query_embedding, k)


    # 3. Map indices back to original data and collect details
    results = []
    # logger.debug("DEBUG, original dataframe is %s", original_df)
    for rank, faiss_id in enumerate(faiss_indices[0]):
        # Use the map to get the original DF index
        original_df_index = indices_map[faiss_id]
        details = original_df.iloc[original_df_index].to_dict()
        # logger.debug("==DEBUG %s", details)
        curr_result = dict()
        curr_result["reactants"] = details["s_reactants"]
        curr_result["products"] = details["s_products"]
        curr_result["solvents"] = details["s_solvents"]
        curr_result["reagents"] = details["s_reagents"]
        if curr_result["reagents"] == curr_result["products"]: continue
        curr_result["experiments_details"] = details["v"]
        curr_result["file_name"] = details.get("file_name", "")
        
        # Add the calculated distance/similarity (L2 distance is returned by default)
        curr_result['distance'] = distances[0][rank]
        curr_result['rank'] = rank + 1
        results.append(curr_result)
        
    return results


