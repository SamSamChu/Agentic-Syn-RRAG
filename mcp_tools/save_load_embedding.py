import pandas as pd
import numpy as np
import faiss
import os
from typing import List, Dict, Any, Tuple

##Log system initialization
import logging
##Basic log configuration: INFO level, includes timestamp, module name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define file paths - absolute paths based on the project root directory
FAISS_INDEX_PATH = os.path.join(_BASE_DIR, 'data', 'reaction_update.faiss')
INDICES_MAP_PATH = os.path.join(_BASE_DIR, 'data', 'indices_map_update.npy')
PD_FILE_PATH = os.path.join(_BASE_DIR, 'data', 'offline_reaction_database.json')

# --- Modified Function to build and save the retrieval index ---

def build_and_save_retrieval_index(embeddings: np.ndarray, indices_map: List[int]):
    """
    Builds an efficient FAISS index and saves it along with the index map to disk.
    """
    faiss.omp_set_num_threads(8)
    #dimension = embeddings.shape
    dimension_size = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension_size)
    # Use IndexFlatL2 for L2 (Euclidean) distance search.
    # index = faiss.IndexFlatL2(dimension)
    
    logger.info("Adding %s embeddings to FAISS index...", embeddings.shape)
    index.add(embeddings)
    logger.info("Index ready. Saving to disk...")


    # Save the FAISS index file
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save the indices map using NumPy's efficient binary format
    np.save(INDICES_MAP_PATH, np.array(indices_map))
    
    logger.info("Saved index to %s and map to %s", FAISS_INDEX_PATH, INDICES_MAP_PATH)


# --- New Function to load the retrieval system from local files ---

def load_retrieval_system() -> Tuple[faiss.Index, List[int], pd.DataFrame]:
    """
    Loads the FAISS index and the indices map from local disk files.
    
    NOTE: The original dataframe ('original_df') is assumed to be loaded separately 
    using standard pandas operations (e.g., pd.read_csv()).
    """
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(INDICES_MAP_PATH):
        raise FileNotFoundError("FAISS index or indices map file not found. Run prepare_offline_data first.")

    logger.info("Loading FAISS index from %s...", FAISS_INDEX_PATH)
    index = faiss.read_index(FAISS_INDEX_PATH)

    logger.info("Loading indices map from %s...", INDICES_MAP_PATH)
    indices_map = np.load(INDICES_MAP_PATH).tolist()

    # Load your original DataFrame here (modify this line to match your file format)
    # df = pd.read_csv('your_reaction_dataset.csv')
    # Since the original dataframe was generated in the previous example, we'll return a placeholder
    # You must load your real data here
    df = pd.read_json(PD_FILE_PATH)
    logger.info("Done, load the json file")
    #df = pd.DataFrame({'reactants_smi': ['placeholder'], 'products_smi': ['placeholder'], 'ExperimentDetails': ['placeholder']})
    # NOTE: Ensure the loaded DF matches the size and indices used during preparation.

    return index, indices_map, df


