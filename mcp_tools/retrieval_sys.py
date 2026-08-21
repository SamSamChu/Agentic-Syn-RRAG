import pandas as pd
import os
import faiss
import asyncio
from database_embedding import prepare_offline_data, retrieve_similar_reactions
from save_load_embedding import build_and_save_retrieval_index, load_retrieval_system

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Filenames for persistence
FAISS_INDEX_PATH = './data/reaction_update.faiss'
INDICES_MAP_PATH = './data/indices_map_update.npy'
from rxngraphormer.rxn_emb import RXNEMB
#INDICES_MAP_PATH = "indices_map.npy"
pretrain_model_path = "./pretrained_classification_model"
rxnemb_calc_pretrained = RXNEMB(pretrained_model_path=pretrain_model_path, model_type="classifier")
# 1. Create a dummy dataset (300k samples represented by 5 here for brevity)
data = {
    'reactants_smi': ['CCO.Br', 'C(=O)O.CN', 'CCC(=O)O.C(=C)C', 'CCOC(=O)C.CCO', 'c1ccccc1.Cl'] * 60000,
    'products_smi': ['CCBr.O', 'CN=C=O.O', 'CCC(=O)OC(=C)C.O', 'CCOC(=O)C(O)C.O', 'c1ccccc1Cl'] * 60000,
    'ExperimentDetails': ['EtOH solvent, 80C, 2h', 'Acetone, RT, overnight', 'Toluene, Reflux, 4h', 'MeOH, 0C, 30min', 'Dichloromethane, RT, UV light'] * 60000
}
# Extend this DataFrame to 300k in your real application
#df = pd.DataFrame(data) 

#df = pd.read_json("./data/cot_distillation_dataset_from_full_dataset_March31.json") #cot_distillation_dataset_from_full_dataset.json")

async def main(df):
    if not os.path.exists(FAISS_INDEX_PATH):
        logger.info("Offline data not found. Preparing and building index...")
        logger.info(f"current indices location {INDICES_MAP_PATH}")
        embeddings, indices_map = prepare_offline_data(df) #prepare_offline_data(df)
        logger.info("START CURRENT embedding")
        build_and_save_retrieval_index(embeddings, indices_map)
        # The faiss_index variable needs to be set after saving for the next step
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        logger.info("Loading index and map from local files...")
        # Note: load_retrieval_system() needs access to the original DF path
        faiss_index, indices_map, df_loaded = load_retrieval_system()
        # Use df_loaded for the retrieval function (assuming it loads correctly)
        df = df_loaded

    logger.debug("DEBUG original dataframe is %s", df)
    # 4. Define a query
    query_r = "CCOC(=O)c1cc2cc(N)c(C(F)(F)F)cc2[nH]c1=O"#'CCC(=O)O.C(=C)C' 
    query_p = "CCOC(=O)c1cc2cc(-n3ccc(C=O)c3)c(C(F)(F)F)cc2[nH]c1=O" #'CCC(=O)OC(=C)C.O' 

    # 5. Retrieve the top 100 results
    top_100_results = await retrieve_similar_reactions(
        query_r, 
        query_p, 
        faiss_index, 
        indices_map, 
        df, 
        k=10
    )

    logger.info("Retrieved %d top candidates.", len(top_100_results))

    # Print the top 3 results
    for i in range(10):
        logger.info("Rank %d (Distance: %.4f):", top_100_results[i]['rank'], top_100_results[i]['distance'])
        #logger.info("Reactants: %s", top_100_results[i]["s_reactants"])
        #logger.info("Products: %s",{top_100_results[i]['s_products']}")
        logger.info("all contents within: %s", top_100_results[i])
        #logger.info("Conditions: %s",{top_100_results[i]['ExperimentDetails']}")
        
if __name__ == "__main__":
    df = pd.read_json("./data/offline_reaction_database.json")
    asyncio.run(main(df))


