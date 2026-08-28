import os
import json
import csv
import re
import asyncio
import litellm
from openai import AsyncOpenAI
from string import Template
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type
import argparse   
from langchain_core.rate_limiters import InMemoryRateLimiter
#define rate limiter
try:
    from dotenv import load_dotenv
    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_PKG_ROOT, ".env"))
except ImportError:
    pass

_API_KEY = os.getenv("TEST_API_KEY")
if not _API_KEY:
    raise RuntimeError(
        "TEST_API_KEY not set. Copy .env.example to .env and fill in your key."
    )
model_id = os.getenv("TEST_JUDGE_MODEL_GPT", "gpt-5-pro")    


rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.08,
    check_every_n_seconds=0.1,
    max_bucket_size=1
)
# Set this to 5-10 depending on your API tier (Free tier use 2-3)
litellm.set_verbose = os.getenv("LITELLM_LOGGING", "").lower() in ("1", "true", "yes")
client = AsyncOpenAI(
    api_key=_API_KEY, # 填入api_key
    base_url=os.getenv("TEST_BASE_URL", "https://www.litellm.org/"), # 设置基础URL
)


prompt_template_gpt = Template("""
Role: You are an expert process chemist acting as a highly critical reviewer
for an automated synthesis planning system. Your task is to evaluate the
full-protocol plausibility of a candidate synthetic procedure.

Target product SMILES:
${product_smiles}

Ground-truth patent procedure, if available:
${patent_data}

Candidate generated protocol:
${agent_recipe}

Evaluation task:
Assign the candidate protocol a full-protocol plausibility score from 0 to 10.
Use the patent procedure as the primary reference when available. If the patent
procedure is incomplete or unavailable, use established chemical knowledge for
the reaction class. Do not reward verbosity, generic laboratory language, or
unjustified numerical precision.

Score the protocol based on the following four aspects:

1. Temperature and time:
Evaluate whether the reaction temperature and duration are explicitly stated,
chemically reasonable, and consistent with the patent or with established
practice. Penalize vague, missing, impossible, or contradictory conditions.

2. Quantities and stoichiometry:
Evaluate whether equivalents, amounts, or concentrations are chemically
plausible and sufficient for the proposed transformation. Penalize missing key
amounts, contradictory limiting reagents, unjustified stoichiometric use of
catalysts, or stoichiometry that would prevent the reaction.

3. Solvent compatibility and concentration:
Evaluate whether the solvent system is specified, chemically compatible with
the reactants and reagents, and used at a plausible volume or concentration.
Penalize missing solvents, incompatible solvents, or conditions such as heating
above the solvent boiling point without pressure control.

4. Overall practicality and safety:
Evaluate whether the procedure is practically executable, including order of
addition, atmosphere or temperature control when needed, monitoring, quench,
work-up, isolation, and safety considerations. Penalize destructive work-up,
unsafe operations, internally contradictory steps, or procedures that are not
actionable.

Hard caps:
- Any major patent mismatch or major chemical implausibility: score ≤ 6.
- Multiple major issues or likely reaction failure: score ≤ 4.
- Fundamentally unsafe or unworkable procedure: score ≤ 2.

Scoring interpretation:
- 9–10: Patent-grade or publication-ready operational detail.
- 6–8: Chemically plausible but incomplete or weakly justified.
- 3–5: Significant operational gaps or risky to reproduce.
- 0–2: Not practically executable.

Return only a JSON object in the following format:

{
  "full_protocol_plausibility_score": <float from 0 to 10>,
  "reasoning": "<Concise explanation of the score, explicitly stating where points were lost>"
}
""")


async def judge_with_litellm_self_correction(product_smiles, patent_data, agent_recipe, sample, model_id="gpt-5-pro"):
    prompt = prompt_template_gpt.substitute(
        product_smiles=product_smiles,
        patent_data=patent_data,
        agent_recipe=agent_recipe
    )
    await rate_limiter.aacquire()
    # LiteLLM Unified Call
    print(f"CURRENT PROMPT: {prompt}")

    response = await client.responses.create(
        model=model_id,
        # Messages are passed as 'input' in this API
        input=[
            {"role": "system", "content": "You are a robotic, highly critical chemistry auditor."},
            {"role": "user", "content": prompt}
        ],
        reasoning={
            "effort": "high",   # GPT-5 Pro defaults to and only supports 'high'
            "summary": 'detailed'     # Request a summary of the internal reasoning
        }
        # Note: 'temperature' is often restricted in reasoning models; 
        # check model docs if this causes an error.
    )



    # 1. Access the usage object
    usage = response.usage
    print(f"DEBUG current response is {response}")

    # 2. Get the breakdown
    full_text = ""
    thinking_text = ""
    
        
    if hasattr(response, "output"):
        completion_tokens = usage.output_tokens
        # Note: For GPT-5-Pro, it's often under output_tokens_details.reasoning
        thinking_tokens = getattr(usage.output_tokens_details, "reasoning", 0)
        
        print(f"Completion Tokens: {completion_tokens}")
        print(f"Thinking Tokens: {thinking_tokens}")
        
        # NEW RESPONSES API (GPT-5-Pro)
        for item in response.output:
            # 1. Handle the actual text response
            if item.type == "message":
                # item.content is a list of ContentBlock objects
                for content_block in item.content:
                    if content_block.type in ("text", "output_text"):
                        # ACCESS THE TEXT FIELD DIRECTLY
                        full_text += content_block.text
            
            # 2. Handle the reasoning/thinking process
            elif item.type == "reasoning":
                if getattr(item, "summary", None):
                    parts = []
                    for s in item.summary:
                        parts.append(getattr(s, "text", "") or str(s))
                    thinking_text += "\n\n".join(parts)
                elif getattr(item, "content", None):
                    thinking_text += str(item.content)

    else:
        completion_tokens = usage.completion_tokens #output_tokens
        print(f"after calling API finished, output the corresponding completion_tokens \n {completion_tokens} \n") 
        thinking_tokens = getattr(usage.output_tokens_details, "reasoning_tokens", 0)
        print(f"after calling API finished, display the corresponding thinking_tokens \n {thinking_tokens} \n")
        full_text = response.choices[0].message.content
        
        thinking_text = getattr(response.choices[0].message, "reasoning_content", "")
        print(f"after calling API finished, display the corresponding thinking_text \n {thinking_text} \n")
        if "gpt" in model_id:
            for item in response.output:
                if item.type == "reasoning":
                    thinking_text = item.summary  # This is your 'thinking_text'
                    break
        print(f"after calling API finished, display the corresponding thinking_text \n {thinking_text} \n")
  

    # Extract JSON
    try:
        # Regex handles cases where LLM puts text before/after the JSON
        json_str = re.search(r"\{.*\}", full_text, re.DOTALL).group()
        scores = json.loads(json_str)
        # Preserve the CoT for the CSV
        scores['full_chain_of_thought'] = thinking_text
        scores["full_text"] = full_text
        scores['inputs'] = sample
        scores["idx"] = sample["idx"]
        print("===debug SHAN: score is ", scores)
        return scores
    except Exception as e:
        print(f"Parsing Error: {e}")
        return None

def save_audit_to_csv(data_ls, prefix="gpt-5-pro_litellm_chemistry_audit"):
    if not data_ls: return
    json_file = prefix + ".json"
    csv_file = prefix + ".csv"
    with open(json_file, 'a', encoding='utf-8') as f:
        # indent=4 adds the visual structure
        # ensure_ascii=False keeps chemical SMILES and symbols readable
        json.dump(data_ls, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved visually formatted JSON to {csv_file}")
    #print(f"FINAL DEBUG result is {data_ls}")
    file_exists = os.path.isfile(csv_file)
    # Use the keys from the first valid dictionary
    fieldnames = data_ls[0].keys()
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header ONLY if the file is brand new
        if not file_exists:
            writer.writeheader()
        
        # 3. FIX SPEED: Write all rows at once
        writer.writerows(data_ls)

# --- Execution ---
async def loop_llm_as_a_judge(samples, model_id, prefix="", prefix_checkpoint="top3_"):
    """
    samples should contains 3 key elements:
    1. product_smiles
    2. patent_data with all key entities as well as details procesure
    3. agent recipe, final full recipe of the agentic's output extraction.
    model_id should be the SOTA model utized for llm_as_judge, from litellm
    prefix, the prefix name for the final csv results outputs
    """
    batch_size = 2 #to limit the total number of current batch
    checkpoint_file = f"{prefix_checkpoint}_checkpoint.json"
    #start_idx = 56
    num_samples = len(samples)
    # 1. LOAD CHECKPOINT
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
            start_idx = checkpoint.get("last_index", 0)
            print(f"🔄 Resuming from checkpoint at index {start_idx}")
    else:
        start_idx = 0 # Or your desired default starting point
        print(f"🆕 Starting fresh run from index {start_idx}")
    while start_idx  < num_samples:
        end_idx = min(start_idx + batch_size, num_samples)
        batch = samples[start_idx:end_idx]

        try:
            # RETRY LOGIC FOR THE BATCH
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(5),
                wait=wait_exponential(multiplier=1, min=4, max=60),
                reraise=True
            ):
                with attempt:
                    print(f"🚀 Processing batch {start_idx}-{end_idx} (Attempt {attempt.retry_state.attempt_number})")

                    tasks = [
                        judge_with_litellm_self_correction(
                            s["product_SMILES"], s["patent_ground_truth"],
                            s["agent_recipe"], s,
                            model_id=model_id
                        ) for s in batch
                    ]

                    results = await asyncio.gather(*tasks)

                    # 2. SAVE RESULTS & UPDATE CHECKPOINT
                    filename_output = f"{prefix}_litellm_chemistry_audit"
                    save_audit_to_csv(results, prefix=filename_output)

                    # Increment index and save progress to JSON
                    start_idx += batch_size
                    with open(checkpoint_file, "w") as f:
                        json.dump({"last_index": start_idx}, f)

            print(f"✅ Batch complete. Checkpoint updated to {start_idx}.")

        except Exception as e:
            print(f"❌ Batch starting at {start_idx} failed permanently: {e}")
            break

    if start_idx >= num_samples:
        print("🏁 All samples processed. Deleting checkpoint.")
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

    

import json
import pandas as pd
def score_all_samples(file_path="gpt-5-pro_litellm_chemistry_audit.csv"):
    df = pd.read_csv(file_path)
    print(f"DEBUG current result is {df}")
    #calculate the mean score for baseline, agentic, and flesh_score
    col = "full_protocol_plausibility_score"
    scores = pd.to_numeric(df[col], errors="coerce")
    valid_scores = scores.dropna()
    print(f"n={len(valid_scores)}, mean={valid_scores.mean():.3f}, std={valid_scores.std():.3f}")
    print(f"min={valid_scores.min():.2f}, max={valid_scores.max():.2f}, median={valid_scores.median():.3f}")
    return valid_scores
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load corresponding files and execute/save related files for LLM-as-a-judge")
    # Define two positional arguments of type int
    parser.add_argument("--input_file", type=str, help="input file to load")
    parser.add_argument("--prefix", type=str, help="prefix filename to save corresponding files")
    args = parser.parse_args()
    #load all samples, a pair of file to read and to save to
    curr_file = args.input_file
    curr_prefix = args.prefix
    #load all samples
    with open(curr_file, 'r') as file: #'llama3_top1_llm_judge_eval_results.json', 'r') as file:
        data_list = json.load(file)
    #run llm_as_a_judge test
    #loop_llm_as_a_judge(data_list)
    prefix = "top1_gpt5-pro_update"
    asyncio.run(loop_llm_as_a_judge(data_list, model_id=model_id, prefix=curr_prefix, prefix_checkpoint=curr_prefix))
    if os.path.exists(filename_output_csv):
        score_all_samples(filename_output_csv)
    else:
        print(f"跳过汇总：找不到 {filename_output_csv}")

