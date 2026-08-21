import asyncio
import json
import logging
import re
import os
import pandas as pd
import ast
import litellm
from openai import AsyncOpenAI
from string import Template
from aiolimiter import AsyncLimiter
import argparse
from datetime import datetime

##Basic log configuration: INFO level, includes timestamp, module name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
try:
    from dotenv import load_dotenv

    _PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_PKG_ROOT, ".env"))
except ImportError:
    pass

# 5 requests per 60 seconds
limiter = AsyncLimiter(5, 60)
# Set this to 5-10 depending on your API tier (Free tier use 2-3)
sem = asyncio.Semaphore(5)  
litellm.set_verbose = True
client = AsyncOpenAI(
    api_key=os.getenv("SOTA_API_KEY"),
    base_url="https://www.litellm.org/" # 设置基础URL
)

# Configure API settings
#os.environ["LITELLM_LOGGING"] = "True"  # Set to True for debugging
model_id = os.getenv("SOTA_MODEL", "gemini-2.5-pro")
BASE_URL = "https://www.litellm.org/"

#prompt_org = #ChatPromptTemplate.from_template
prompt_org = Template("""
        You are a chemistry expert. Given the canonical target product molecule(s): ${product_smiles}
        
        Reasoing first include:
            Analyze the product molecule(s) firstly, and then secondly propose strating materials rectants. Thirdly identify reactions conditions include all reagents and solvents. 
        This reaction pathway is only for single step retro-synthesis. Be decisive and recommend one valid pathway for single-step retro-synthesis.
        ALWAYS REASON FIRST before give the correponding JSON.
        Finally propose whole reaction recipes on JSON format include: 
            1. recommendation_and_ranking (with rational, ranking, confidence score(1-5, based on whole recipes).).   
            2. reaction_name     
            3. All organic synthesis reaction_conditions inculde reactants, reagents, solvents, temperature, quantity recommendation of every component (specific number of temperature in unit of °C, specific number of reaction time in units of hours, quantity recommendation with equivalents, solvents' volume in unit of ML, every component include name, smiles, quantity),
            4. procedure.
        Final JSON should be with key words include: reaction_name, recommendation_and_ranking, reaction_conditions(inner keys include reactants (with keys: name, smiles, equivalents), reagents (with keys: name, smiles, equivalents), solvents (with keys: name, smiles, volume), temperature, time), procedure.
        """)

async def litellm_self_generation(product_smiles, model_id=None):
    prompt_template = Template("""
    Role: You are an Expert Process Chemist acting as a Reviewer for an automated synthesis planning system. Your goal is to evaluate two candidate AI recipes against a Ground Truth Patent Recipe.

    # Target molecule(s) products SMILES:${product_smiles}

    
    # Output Format
    Provide your response strictly as a JSON object:

    ```json
    {
    "local_bones_score": <float 0-10>,
    "local_bones_reasoning": "<Concise explanation of chemical identity for Local>",
    "agentic_bones_score": <float 0-10>,
    "agentic_bones_reasoning": "<Concise explanation of chemical identity for Agentic>",
    "agentic_flesh_score": <float 0-10>,
    "agentic_flesh_reasoning": "<In your JSON `agentic_flesh_reasoning`, you must explicitly state WHERE the points were lost (e.g., "Lost 1pt on Temp (Generic 'Reflux' vs Patent '80C'); Lost 1pt on Workup (Missing extraction solvent).").
>",
    "is_agentic_optimal": <bool>,
    "optimization_reasoning": "<If true, explain why Agentic is better than Patent. If false, write 'None'>"
    }""")

    prompt = prompt_org.substitute(
        product_smiles=product_smiles,

    )
    # LiteLLM Unified Call
    print(f"CURRENT PROMPT: {prompt}")
    async with limiter:
        response = await client.chat.completions.create(
                    model=model_id,
                    stream=False,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=1, # for Gemini with temperature 0, clause temperatue 1 for thinking
                    reasoning_effort="high", 
                    response_format={"type": "json_object"}
                )

        full_text = response.choices[0].message.content
        
        # 1. Access the usage object
        usage = response.usage

        # 2. Get the breakdown
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        print(f"after calling API finished, output the corresponding completion_tokens \n {completion_tokens} \n")  
        thinking_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0)
        print(f"after calling API finished, display the corresponding thinking_tokens \n {thinking_tokens} \n")
        thinking_text = getattr(response.choices[0].message, "reasoning_content", "")
        print(f"after calling API finished, display the corresponding thinking_text \n {thinking_text} \n")
    

        # Extract JSON
        try:
            #pattern = r"```json\s*(\{.*?\})\s*```"
            #match = re.search(pattern, full_text, re.DOTALL)
            #json_str = match.group(1).strip()
            #results = json.loads(json_str)
            json_str = re.search(r"\{.*\}", full_text, re.DOTALL).group()
            results = json.loads(json_str)
            # Preserve the CoT for the CSV
            results['full_chain_of_thought'] = thinking_text
            results['full_text'] = full_text
            print("===debug SHAN: score is ", results)
            return results
        except Exception as e:
            print(f"Parsing Error: {e}")
            results = dict()
            results['full_chain_of_thought'] = thinking_text
            results['full_text'] = full_text
            return results

import pickle
#to_save_pkl = "gemini2_5_whole_recipes_onlyNative.pkl"
#to_save = "gemini2_5_whole_recipes_onlyNative.json"
async def main(prefix_checkpoint="tmp_", to_input_csv="evaluate/500_samples_March31_6_rag_agentic.csv", to_save_prefix="gemini2_5_whole_recipes_onlyNative", start_index=0, end_index=None,):
    logger.info(
        "模型=%s | 时间=%s",
        os.getenv("SOTA_MODEL", "gemini-2.5-pro"),  
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    to_save = f"{to_save_prefix}.csv"
    to_save_pkl = f"{to_save_prefix}.pkl"
    sample_inputs = [{
        "messages": [("user", "What are the common conditions for this reaction?")],
        "products_SMILES": "Cl.CS(=O)(=O)C1CCCNC1", #"C1C=CCC2=CC=CC=C12", #"CC1=CC=C(C=C1)C#CC#CC(C)(C)O", #"O=Cc1cn(C(=O)OCCl)c2ccccc12",#"CCOC(=O)c1cc2cc(N)c(C(F)(F)F)cc2[nH]c1=O",#"CCOC(=O)c1cc2cc(-n3ccc(C=O)c3)c(C(F)(F)F)cc2[nH]c1=O"
    },
                """{"messages": [("user", "What are the common conditions for this reaction?")],
        "products_SMILES": "COc1cncc2c1[C@]1(O)[C@H](O)[C@H](C(N)=O)[C@@H](c3ccccc3)[C@]1(c1ccc(Br)cc1)O2",#"CCOC(=O)c1cc2cc(N)c(C(F)(F)F)cc2[nH]c1=O",#"CCOC(=O)c1cc2cc(-n3ccc(C=O)c3)c(C(F)(F)F)cc2[nH]c1=O"
    }"""
                        ]
    count =0 
    test_500 = pd.read_csv(to_input_csv) #500_samples_rag_agentic.csv")
    sample_inputs = []
    start_index = start_index
    if end_index is None:
        end_index = len(test_500)
    #read checkpoint file
    checkpoint_file = f"{prefix_checkpoint}_checkpoint.json"
    #start_idx = 56
    # 1. LOAD CHECKPOINT
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
            start_idx = checkpoint.get("last_index", 0)
            print(f"🔄 Resuming from checkpoint at index {start_idx}")
    else:
        start_idx = start_index # Or your desired default starting point
        print(f"🆕 Starting fresh run from index {start_idx}") 
    for row in test_500.itertuples(): #for inputs in sample_inputs:
        #print(f"current row is {row} and products canonical final one is {row.s_reactants}")
        assert count <= end_index
        products_s = row.s_products
        reactants_s = row.s_reactants
        solvents_s = row.s_solvents
        reagents_s = row.s_reagents
        #print(f"DEBUG SHAN clean reponse part is {row.clean_response} and type is {type(row.clean_response)}")
        data = ast.literal_eval(row.clean_response)
        count += 1
        if count <= start_idx: continue
        if end_index is not None and count > end_index:
            break
        final_recipes = await litellm_self_generation(products_s, model_id=model_id)
        entry = {
            "idx": count,   #the index among all 500 samples
            "patent_products": products_s,
            "patent_reactants":reactants_s,
            "patent_solvents": solvents_s,
            "patent_reagents": reagents_s,
            "patent_key":row.k,
            "patent_details":row.v,
            "full_recipe": final_recipes,
        }
        with open(to_save_pkl, "ab") as f:
            pickle.dump(entry, f)
        with open(to_save, "a") as f:
            #f.write(json.dumps(entry) + "\n")
            json.dump(entry, f, indent=4)
            f.write("\n")
        with open(checkpoint_file, "w") as f:
            json.dump({"last_index": count}, f)
        print(f"✅ Checkpoint updated to {count}")
    
# Entry point to run the async code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple addition script")
    parser.add_argument("-f", "--input_file", type=str, help="the input csv file to process to get the corresponding reaction protocal")
    parser.add_argument("-s", "--output_file", type=str, help="the output json and pkl file prefix to save all related output")
    parser.add_argument("-p", "--prefix", type=str, default="native_")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(main(
        prefix_checkpoint=args.prefix,
        to_input_csv=args.input_file,
        to_save_prefix=args.output_file,
        start_index=args.start_index,
        end_index=args.end_index,
    ))
    


