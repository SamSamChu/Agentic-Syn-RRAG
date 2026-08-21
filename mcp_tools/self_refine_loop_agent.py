from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import asyncio
import operator
from langgraph.graph.message import add_messages
from rdkit import Chem
from openai import OpenAI
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from collections import Counter
from pathlib import Path
import sys

##Log system initialization
import logging
import os
##Basic log configuration: INFO level, includes timestamp, module name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.reaction_plausibility import check_reactant_scaffold_conservation

LLM_SERVER_URL = os.environ.get("LLM_SERVER_URL", "http://localhost:8000/mcp")

# Initialize globally to keep the connection and model persistent
mcp_client = MultiServerMCPClient({
    "SynthesisServer": {
        "url": LLM_SERVER_URL,
        "transport": "http"
    }
})


def update_top_results(current: List[dict], new_vals: List[dict]) -> List[dict]:
    combined = current + new_vals
    return sorted(combined, key=_candidate_sort_key)

class SynthesisState(TypedDict):
    messages: Annotated[list, add_messages]
    product_smiles: str
    verified_results: Annotated[list, operator.add]
    conditions_results: Annotated[list, operator.add]
    is_valid: bool
    iterations: int
    num_samples: int
    # This key will always hold the top 3 passed syntheses
    top_5_results: Annotated[List[dict], update_top_results]

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
client = OpenAI(base_url=LLM_BASE_URL, api_key="token")


# def _beam_schedule(num_samples: int) -> List[int]:
#     """Run greedy plus milestone beam sizes up to the requested top-k."""
#     requested = max(1, int(num_samples or 1))
#     schedule = [beam for beam in (1, 3, 5, 10) if beam <= requested]
#     if requested not in schedule:
#         schedule.append(requested)
#     return sorted(set(schedule))

def _beam_schedule(num_samples: int) -> List[int]:
    return [max(1, int(num_samples or 1))]

def _canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _split_components(smiles: str) -> List[str]:
    return [part for part in (smiles or "").split(".") if part]


def _element_counts(smiles: str) -> Counter:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return Counter()
    return Counter(
        atom.GetSymbol()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    )


def _element_consistent(reactants_smiles: str, product_smiles: str) -> bool:
    reactant_elements = set(_element_counts(reactants_smiles))
    product_elements = set(_element_counts(product_smiles))
    return bool(product_elements) and product_elements.issubset(reactant_elements)


def _scaffold_valid(reactants_smiles: str, product_smiles: str) -> bool:
    try:
        result = check_reactant_scaffold_conservation(
            _split_components(reactants_smiles),
            product_smiles,
            timeout=3,
        )
        return bool(result.get("valid", False))
    except Exception as exc:
        logger.warning("Scaffold validation failed for %s: %s", reactants_smiles, exc)
        return False


def _abnormal_penalty(reactants_smiles: str, product_smiles: str) -> float:
    reactant_mol = Chem.MolFromSmiles(reactants_smiles or "")
    product_mol = Chem.MolFromSmiles(product_smiles or "")
    if reactant_mol is None or product_mol is None:
        return 100.0

    product_heavy = max(1, product_mol.GetNumHeavyAtoms())
    reactant_heavy = reactant_mol.GetNumHeavyAtoms()
    fragments = Chem.GetMolFrags(reactant_mol, asMols=True, sanitizeFrags=True)
    fragment_count = len(fragments)
    largest_fragment = max((frag.GetNumHeavyAtoms() for frag in fragments), default=0)

    penalty = 0.0
    if fragment_count > 4:
        penalty += (fragment_count - 4) * 1.5
    if reactant_heavy > product_heavy * 2.5:
        penalty += (reactant_heavy / product_heavy - 2.5) * 4.0
    if largest_fragment > product_heavy * 1.8:
        penalty += (largest_fragment / product_heavy - 1.8) * 4.0

    product_counts = _element_counts(product_smiles)
    reactant_counts = _element_counts(reactants_smiles)
    for symbol, count in reactant_counts.items():
        product_count = product_counts.get(symbol, 0)
        if symbol in {"C", "Si", "B", "P", "S"} and product_count:
            allowed = max(product_count * 3, product_count + 8)
            if count > allowed:
                penalty += (count - allowed) * 0.5
        elif product_count == 0 and symbol not in {"Cl", "Br", "I", "F", "Na", "K", "Li", "Cs"}:
            penalty += count * 0.25

    return penalty


def _candidate_sort_key(candidate: dict):
    return (
        not candidate.get("forward_consistent", candidate.get("is_consistent", False)),
        not candidate.get("scaffold_valid", False),
        not candidate.get("element_consistent", False),
        candidate.get("abnormal_penalty", 100.0),
        candidate.get("source_priority", 999),
        candidate.get("source_rank", 999),
        candidate.get("smiles", candidate.get("reactants", "")),
    )


def _score_from_candidate(candidate: dict) -> float:
    score = 0.0
    if candidate.get("forward_consistent", candidate.get("is_consistent", False)):
        score += 100.0
    if candidate.get("scaffold_valid", False):
        score += 10.0
    if candidate.get("element_consistent", False):
        score += 5.0
    score -= float(candidate.get("abnormal_penalty", 0.0))
    score -= float(candidate.get("source_priority", 999)) * 0.01
    score -= float(candidate.get("source_rank", 999)) * 0.001
    return score


async def generate_reactants_fn(state: SynthesisState):
    product_smiles = state["product_smiles"]
    
    # 1. Fetch the tools from the remote MCP server
    # This is fast because the model is already loaded on the server
    tools = await mcp_client.get_tools()
    
    # 2. Find your specific SFT tool by name
    # Ensure this matches the name in your @mcp.tool() definition
    sft_tool = next(t for t in tools if t.name == "generate_synthesis_results")
    # Construct prompt with feedback if this is a retry
    prompt = f"You are Expert in organic chemical reaction. Translate product into reactant using the language of retrosynthesis. Respond with SMILES strings. Suggest synthetic route to target molecules with SMILES: {product_smiles} focusing on efficiency. Recommend reactants:"

    # 3. Call the SFT model via MCP
    # We use ainvoke for asynchronous tool execution, for num_samples being 3

    raw_candidates = []
    failed_this_round = []

    for source_priority, beam_size in enumerate(_beam_schedule(state["num_samples"])):
        prediction_result = await sft_tool.ainvoke({
            "prompts": [prompt],
            "num_samples": beam_size,
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 65,
        })
        for source_rank, result in enumerate(prediction_result):
            sm = result["text"].strip()
            raw_candidates.append({
                "raw_smiles": sm,
                "beam_size": beam_size,
                "source_priority": source_priority,
                "source_rank": source_rank,
            })

    canonical_candidates = {}
    for item in raw_candidates:
        canonical = _canonical_smiles(item["raw_smiles"])
        if canonical is None:
            failed_this_round.append(item["raw_smiles"])
            continue

        existing = canonical_candidates.get(canonical)
        if existing is None:
            candidate = {
                "smiles": canonical,
                "raw_smiles": item["raw_smiles"],
                "source_priority": item["source_priority"],
                "source_rank": item["source_rank"],
                "source_beams": [item["beam_size"]],
                "scaffold_valid": _scaffold_valid(canonical, product_smiles),
                "element_consistent": _element_consistent(canonical, product_smiles),
                "abnormal_penalty": _abnormal_penalty(canonical, product_smiles),
                "forward_consistent": False,
            }
            candidate["score"] = _score_from_candidate(candidate)
            canonical_candidates[canonical] = candidate
        else:
            existing["source_beams"].append(item["beam_size"])
            if (item["source_priority"], item["source_rank"]) < (
                existing["source_priority"],
                existing["source_rank"],
            ):
                existing["source_priority"] = item["source_priority"]
                existing["source_rank"] = item["source_rank"]
                existing["raw_smiles"] = item["raw_smiles"]
                existing["score"] = _score_from_candidate(existing)

    valid_candidates = sorted(canonical_candidates.values(), key=_candidate_sort_key)

    # Update state: The reducer on top_5_results will handle the "keep best 3" logic
    logger.info(
        "Generated %d raw candidates, %d valid canonical candidates with beam schedule %s",
        len(raw_candidates),
        len(valid_candidates),
        _beam_schedule(state["num_samples"]),
    )
    
    return {
        "top_5_results": valid_candidates,
        "failed_attempts": state.get("failed_attempts", []) + failed_this_round,
        "is_valid": len(valid_candidates) > 0,
        "iterations": state.get("iterations", 0) + 1
    }

    
async def check_reactants_fn(state: dict):
    candidate = state["candidate"]
    target = state["target"]
    condition = state["condition"]
    metadata = state.get("metadata", {})
    
    # 1. Fetch the tools from the remote MCP server
    # This is fast because the model is already loaded on the server
    tools = await mcp_client.get_tools()
    
    # 2. Find your specific SFT tool by name
    # Ensure this matches the name in your @mcp.tool() definition
    sft_tool = next(t for t in tools if t.name == "generate_synthesis_results")
    
    # 3. Construct the prompt and call the tool
    #prompt = f"Expert in organic chemical reaction. Run synthetic movie forward. Speak only SMILES. Reactants: {candidate}, what product(s) does this reaction result in?"
    prompt = f"Mode: Chemical pathway designer.  === Forward organic synthetic predicts === Output the SMILES of the required reactants. When we have reactants like  {candidate} and {condition}, what product(s) does this reaction load to?"
    # Invoke the tool (this sends the request to your HF-backend server)
    prediction_result = await sft_tool.ainvoke({"prompts": [prompt]})
    # every time only one of the conditions be generated
    prediction = prediction_result[0]["text"].strip()
    
    # Canonicalize SMILES using RDKit for exact comparison
    def canonical(s): 
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else None

    is_match = canonical(prediction) == canonical(target)
    result = {
        **metadata,
        "reactants": candidate,
        "is_consistent": is_match,
        "forward_consistent": is_match,
        "prediction": prediction,
        "condition": condition,
    }
    result["score"] = _score_from_candidate(result)
    
    return {
        "verified_results": [result]
    }
    

def fan_out_logic(state: SynthesisState):
    # Trigger check_consistency_fn for every candidate found
    return [
        Send("verify_consistency", {
            "candidate": c["reactant"],
            "condition": c["output"],
            "target": state["product_smiles"],
            "metadata": c.get("metadata", {}),
        })
        for c in state["conditions_results"]
    ]
    
def aggregate_results(state: SynthesisState):
    # This node is reached AFTER all 'generate_conditions' tasks are done.
    
    # 1. Grab everything collected by the workers
    all_conditions = state.get("conditions_results", [])
    all_verified = state.get("verified_results", [])
    
    # logger.debug("DEBUG: Collected %d conditions, %s", len(all_conditions), all_conditions)
    
    sorted_verified = sorted(all_verified, key=_candidate_sort_key)
    sorted_verified = sorted_verified[: max(1, int(state.get("num_samples", len(sorted_verified)) or 1))]
    
    # IMPORTANT: In LangGraph, if you want to REPLACE the list instead of 
    # ADDING to it (since you are sorting/filtering), you often need to 
    # return a new key or use a 'replace' reducer.
    
    # For now, let's just make sure we aren't clearing conditions:
    return {
        "verified_results": {"__overwrite__": sorted_verified}, # Caution: this will APPEND if using operator.add
    }


async def generate_conditions_fn(state: dict):
    reactant = state["reactant"]
    product = state["product"]
    metadata = state.get("metadata", {})
    
    # 1. Fetch the tools from the remote MCP server
    # This is fast because the model is already loaded on the server
    tools = await mcp_client.get_tools()
    
    # 2. Find your specific SFT tool by name
    # Ensure this matches the name in your @mcp.tool() definition
    sft_tool = next(t for t in tools if t.name == "generate_synthesis_results")
    # Use extra_body for vLLM-specific top_k
    #You are Expert in organic chemical reaction. To translate reactant into products using the language of organic synthesis, determine generation conditions. Respond with SMILES strings. Given reactant SMILES is COC(=O)C(C)(C)C[C@@H]1Cc2ccc(OCc3c(F)cccc3Cl)cc2N(S(=O)(=O)c2ccc(F)c(OC)c2)C1 and product SMILES is COc1cc(S(=O)(=O)N2C[C@H](CC(C)(C)C(=O)O)Cc3ccc(OCc4c(F)cccc4Cl)cc32)ccc1F. Identify solvents and reagents.
    content = f"You are Expert in organic chemical reaction. To translate reactant into products using the language of organic synthesis, determine generation conditions. Respond with SMILES strings. Given reactant SMILES is {reactant} and product SMILES is {product}. Identify solvents and reagents."
    #native = [{"role":"user", "content": content}]
    # 3. Call the SFT model via MCP
    # Note: We pass parameters like temperature/top_p through the MCP tool 
    # if your server-side @mcp.tool supports them.
    prediction_result = await sft_tool.ainvoke({
        "prompts": [content],
        "temperature": 0.3,
        "top_p": 0.8
    })
    
    # logger.debug("DEBUG current response is %s", prediction_result)
    # Return result to be added to the global conditions_results list, every time every thread to generate one condtion
    return {
        "conditions_results": [{
            "reactant": reactant,
            "output": prediction_result[0]["text"].strip(),
            "metadata": metadata,
        }]
    }

# The router that fans out the work
def continue_to_conditions(state: SynthesisState):
    # This creates a parallel 'thread' for every verified reactant
    return [
        Send("generate_conditions", {
            "reactant": r["smiles"], 
            "product": state["product_smiles"],
            "metadata": r,
        }) 
        for r in state["top_5_results"]
    ]

def final_verify(state: SynthesisState):
    pass
   
def self_refine_loop():
    workflow = StateGraph(SynthesisState)

    # Define nodes as Python functions
    workflow.add_node("generate_reactants", generate_reactants_fn)
    workflow.add_node("verify_consistency", check_reactants_fn)
    workflow.add_node("generate_conditions", generate_conditions_fn)
    workflow.add_node("aggregate", aggregate_results)
    #workflow.add_node("final_verify", final_verify_fn)

    # Define Edges
    workflow.add_edge(START, "generate_reactants")
    # Conditional edge for fan-out (Parallel Map)
    # workflow.add_conditional_edges("generate_reactants", fan_out_logic, ["verify_consistency"])
    # Use a conditional edge to trigger the parallel 'Send'
    #workflow.add_conditional_edges(
    #    "verify_consistency", # The node that finishes before the loop
    #    continue_to_conditions,
    #    ["generate_conditions"]
    #)
    # change the order of conditions generation and verification
    # Conditional edge for fan-out (Parallel Map)
    # Use a conditional edge to trigger the parallel 'Send'
    workflow.add_conditional_edges(
        "generate_reactants", # The node that finishes before the loop
        continue_to_conditions,
        ["generate_conditions"]
    )
    workflow.add_conditional_edges("generate_conditions", fan_out_logic, ["verify_consistency"])
    # After all parallel nodes finish, they automatically converge to the next node
    # workflow.add_edge("generate_conditions_node", "final_aggregator_node")
    # workflow.add_edge("generate_conditions", END)
    # Automatic fan-in (Reduce)
    workflow.add_edge("verify_consistency", "aggregate")
    #workflow.add_edge("generate_conditions", "aggregate")
    workflow.add_edge("aggregate", END)

    app = workflow.compile()
    
    return app

async def main():
    app = self_refine_loop()
    # image_data = app.get_graph().draw_png()
    # with open("refine_loop_agent_graph.png", "wb") as f:
    #     f.write(image_data)

    samples = [
        {"messages": [("user", "What are the common conditions for this reaction?")],
        "product_smiles": "CN(CCCCCCCCCN1CCC(OC(=O)Nc2ccccc2-c2ccccc2)CC1)Cc1cccc(O)c1Cl.O=S(=O)(O)c1cccc2c(S(=O)(=O)O)cccc12",
         "num_samples": 3,
        "verified_results": [] # Initialize the list for the reducer
        },
        {
        "messages": [("user", "What are the common conditions for this reaction?")],
        "product_smiles": "COc1cncc2c1[C@]1(O)[C@H](O)[C@H](C(N)=O)[C@@H](c3ccccc3)[C@]1(c1ccc(Br)cc1)O2",
        "num_samples": 3,
        "verified_results": [] # Initialize the list for the reducer 
        }, #O=C(Nc1ccc(S(=O)(=O)F)cc1)c1cc(Cl)ccc1[N+](=O)[O-]
        {
        "messages": [("user", "What are the common conditions for this reaction?")],
        "product_smiles": "O=C(Nc1ccc(S(=O)(=O)F)cc1)c1cc(Cl)ccc1[N+](=O)[O-]",
        "num_samples": 1,
        "verified_results": [] # Initialize the list for the reducer 
        }
        ]

    logger.info("--- Starting Organic Synthesis Agent Loop for 500 samples ---")    
    
    with open("batch_results_self_refine_loop.jsonl", "a", encoding="utf-8") as f:
        for i, inputs in enumerate(samples):
            logger.info(">>> Processing Sample %d/500", i+1)
            
            # Reset state tracker for each new sample
            current_full_state = inputs.copy()
            # Track the state throughout the stream
            #current_full_state = inputs 

            async for chunk in app.astream(inputs, stream_mode="updates"):
                for node_name, update in chunk.items():
                    logger.debug("[Node: %s]", node_name)
                    
                    if node_name == "aggregate":
                        # The aggregate node has already sorted/filtered the data.
                        # We REPLACED our local tracker with this final version.
                        current_full_state.update(update)
                        logger.info("Final Aggregated Count: %d", len(update.get('verified_results', [])))
                    
                    else:
                        # For worker nodes (generate_conditions), we APPEND
                        for key, value in update.items():
                            if key in ["conditions_results", "verified_results"] and isinstance(value, list):
                                if key not in current_full_state:
                                    current_full_state[key] = []
                                current_full_state[key].extend(value)
                            else:
                                current_full_state[key] = value

            logger.info("="*60)
            logger.info("--- FINAL TOP-3 VERIFIED RESULTS ---")
            # logger.debug("Full state: %s", current_full_state)
            logger.info("Verified results: %s", current_full_state["verified_results"])
            # logger.debug("Conditions results: %s", current_full_state["conditions_results"])
            logger.info("="*60)
            # 4. Write the final state of this specific sample immediately
            f.write(json.dumps(current_full_state) + "\n")
            f.flush()  # Force write to disk to prevent data loss on crash

    logger.info("Batch processing complete. Results saved to batch_results.jsonl")
    

# Entry point to run the async code
if __name__ == "__main__":
    asyncio.run(main())

