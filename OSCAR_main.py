from functools import partial
from langgraph.graph import END, StateGraph, START
from typing import Annotated, List, TypedDict, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

##LOAD all related tools:
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId

##define the generation part
import operator
from langgraph.types import Send, Command
from nodes import pathway_generate_only_node, pathway_simple_pipeline_node, fallback_agent_node, pathway_pipeline_node
import asyncio
from mcp.types import Notification
import json
from langchain_core.messages import ToolMessage,AIMessage
import uuid
import json
import re
from rdkit import Chem


##Log system initialization
import logging
##Basic log configuration: INFO level, includes timestamp, module name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PathwayState(TypedDict):
    """The local state for one specific synthesis pathway."""
    pathway_id: int
    reactants: str
    condition: str
    documents: List[str]
    analysis: str  # The LLM reasoning for this specific pathway
    final_output: str # The individual generated snippet

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    reactants_SMILES: str
    products_SMILES: str
    products_IUPAC: str
    num_samples: int
    scaffold_validation: bool
    validation_errors: Annotated[List[Dict], operator.add]
    # NEW: Store the multiple pathways from the Refiner
    # This allows the Parallel Router to "see" what to fan-out
    pathways: List[Dict]
    documents: List[str]
    # This collects all parallel results for the final aggregator
    final_reports: Annotated[List[str], operator.add]
    pathway_id: Annotated[List[int], operator.add]  
    top5_retrieval: Annotated[List[str], operator.add]
    top3_rerank_retrieval: Annotated[List[str], operator.add]
    top3_rerank_extract: Annotated[List[str], operator.add]
    citations: Annotated[List[List[Dict]], operator.add]
    history: Annotated[List[List[str]], operator.add]



# Define how to connect to your MCP server
server_params = StdioServerParameters(
    command="python",
    args=["-u", "./mcp_tools/mcp_rag.py"],
)

def smart_router(state: AgentState):
    if state.get("products_SMILES") is None:
        return "exit"
    return "continue"

def parallel_router(state: AgentState):
    """
    Fires AFTER refine_tools. 
    Sends each of the 3 refined pathways into a parallel RAG+Generate branch.
    """
    # Use .get() to safely retrieve the list updated by the tool
    if state.get("products_SMILES") is None:
        return "exit"
    pathways = state.get("pathways", [])
    logger.info("===current pathways is %s", pathways)
    if not pathways or len(pathways) == 0:
        #return "final_aggregate" # Skip to end if refiner failed
        return "fallback_agent_node"

    return [
        Send("pathway_pipeline", {
            "pathway_id": p["pathway_id"],
            "reactants": p["reactants"],
            "condition": p["condition"],
            "products_SMILES": state["products_SMILES"],
            "scaffold_validation": state.get("scaffold_validation", True),
        }) for p in pathways
    ]

def final_aggregate_node(state: AgentState):
    """Combines all parallel pathway reports into one final response."""
    reports = state.get("final_reports", [])
    val_errors = state.get("validation_errors", [])
    # 1. Split successes and errors based on your prefix
    successes = [
        r for r in reports
        if "PATHWAY ANALYSIS:" in r and "SCAFFOLD VALIDATION WARNING" not in r
    ]
    scaffold_warnings = [r for r in reports if "SCAFFOLD VALIDATION WARNING" in r]
    errors = [r for r in reports if "ERROR for" in r]

    
    # 2. Build the final message content
    sections = []
    
    if successes:
        sections.append("### SUCCESSFUL PATHWAYS\n" + "\n\n".join(successes))
    
    if scaffold_warnings:
        sections.append("### SCAFFOLD VALIDATION WARNINGS\n" + "\n\n".join(scaffold_warnings))

    if errors:
        # We append errors at the bottom so they don't clutter the top
        sections.append("### FAILED PATHWAYS\n" + "\n".join(errors))
        
    if not sections:
        final_content = "No pathways were processed."
    else:
        final_content = f"Synthesis analysis complete ({len(successes)} succeeded, {len(errors)} failed):\n\n" + "\n\n".join(sections)
    
    return {"messages": [AIMessage(content=final_content)]}

@tool
async def verify_and_refine_synthesis(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId], # Mandatory for Command
    mcp_refiner,
    num_samples,
) -> Command:
    """Triggers self-refinement and forces a state update for 'pathways'."""
    # ... your existing extraction logic ...
    logger.info("DDEBUG, START PATHWAY LOCAL designer")
    product_smiles = state.get("products_SMILES")
    result = await mcp_refiner.ainvoke({"product_smiles": product_smiles, "num_samples": num_samples})
    
    # ... your existing JSON parsing logic ...
    refined_list = json.loads(result[0]["text"])["refined_pathways"]
    logger.info("DEBUG refined result after local llm propose %s", result[0])
    formatted_pathways = [
        {
            "pathway_id": i,
            "reactants": p.get("reactants"),
            "condition": p.get("condition"),
            "products_SMILES": product_smiles
        } for i, p in enumerate(refined_list)
    ]

    # --- THE 2026 FIX ---
    return Command(
        update={
            # Update the specific state key so the router can see it
            "pathways": formatted_pathways, 
            # Still update the message history so the LLM stays in the loop
            "messages": [
                ToolMessage(
                    content=f"Successfully refined {len(formatted_pathways)} pathways.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )

async def smart_refine_node(state: AgentState,mcp_refiner,num_samples):
    """Validates SMILES and immediately executes refinement if safe."""
    try:
        # 1. Chemical Validation (The Gatekeeper)
        raw_input = state.get("products_SMILES")
        mol = Chem.MolFromSmiles(raw_input)
        
        if mol is None:
            logger.error(f"Invalid SMILES detected: {raw_input}")
            return {
                "products_SMILES": None, # Signal for the router to exit
                "messages": [AIMessage(content=f"Terminating: '{raw_input}' is not a valid SMILES.")]
            }
        
        # 2. Canonicalization
        canonical_smiles = Chem.MolToSmiles(mol)
        logger.info(f"SMILES validated and canonized: {canonical_smiles}")
        refine_state = dict(state)
        refine_state["products_SMILES"] = canonical_smiles
        # 3. Direct Execution
        # We call the coroutine directly to bypass ToolNode/ainvoke validation issues
        # We pass the state and a generated ID manually
        effective_n = state.get("num_samples") or num_samples
        result_command = await verify_and_refine_synthesis.coroutine(
            state=refine_state, 
            tool_call_id=f"refine_{uuid.uuid4().hex}",
            mcp_refiner=mcp_refiner,num_samples=effective_n
        )

        # 4. Merge canonical SMILES into the final update
        if isinstance(result_command, Command):
            result_command.update["products_SMILES"] = canonical_smiles
            return result_command
        
        return result_command

    except Exception as e:
        logger.exception("Unexpected error in smart_refine_node")
        return {"products_SMILES": None}

def create_chemistry_app(mcp_retrieve, mcp_refiner, num_samples=1, simple_rag=False, generate_only=False,checkpointer=None):
    # --- Define node logic that uses the passed tools ---
    # --- Build the workflow ---
    workflow = StateGraph(AgentState)
    
    # build partial pathway node
    if simple_rag and not generate_only:
        bound_pathway_node = partial(pathway_simple_pipeline_node, mcp_retrieve=mcp_retrieve)
    elif generate_only and not simple_rag:
        bound_pathway_node = partial(pathway_generate_only_node, mcp_retrieve=mcp_retrieve)
    else:
        bound_pathway_node = partial(pathway_pipeline_node, mcp_retrieve=mcp_retrieve)
    bound_smart_refine_node = partial(smart_refine_node,mcp_refiner=mcp_refiner, num_samples=num_samples)
    workflow.add_node("smart_refine", bound_smart_refine_node)
    workflow.add_node("pathway_pipeline", bound_pathway_node)#pathway_pipeline_node)
    workflow.add_node("fallback_agent_node", fallback_agent_node)
    workflow.add_node("final_aggregate", final_aggregate_node)

    # --- Edges ---
    workflow.add_edge(START, "smart_refine")
    workflow.add_conditional_edges(
    "smart_refine", # The node that just finished
    parallel_router, # The function that decides where to go
    {
        "pathway_pipeline": "pathway_pipeline",
        "fallback_agent_node": "fallback_agent_node",
        "exit": END # Add the exit path here!
    })
    workflow.add_edge("pathway_pipeline", "final_aggregate")
    workflow.add_edge("fallback_agent_node", "final_aggregate")
    workflow.add_edge("final_aggregate", END)

    return workflow.compile(checkpointer=checkpointer)

to_save = "whole_recipes_test_paper.json"
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize() 
            # 2. Notify the server you are ready (mandatory for most 2026 SDKs)
            # Without this, some servers close the connection thinking the client hung
            await session.send_notification(Notification(method="notifications/initialized",params={}))
            
            # 2. Load the MCP tools (this includes retrieve_docs)
            mcp_tools = await load_mcp_tools(session)
            logger.info("✅ Tools Loaded: %s", [t.name for t in mcp_tools])
            
            mcp_retrieve = next(t for t in mcp_tools if t.name == "search_similar_reactions")
            # 1. Locate the new tool from the loaded MCP tools list
            mcp_refiner = next(t for t in mcp_tools if t.name == "execute_synthesis_refinement")
            
            # 3. Create the App (Separated logic)
            app = create_chemistry_app(mcp_retrieve, mcp_refiner)

            sample_inputs = [{
                "messages": [("user", "What are the common conditions for this reaction?")],
                "products_SMILES": "Cl.CS(=O)(=O)C1CCCNC1", #"C1C=CCC2=CC=CC=C12", #"CC1=CC=C(C=C1)C#CC#CC(C)(C)O", #"O=Cc1cn(C(=O)OCCl)c2ccccc12",#"CCOC(=O)c1cc2cc(N)c(C(F)(F)F)cc2[nH]c1=O",#"CCOC(=O)c1cc2cc(-n3ccc(C=O)c3)c(C(F)(F)F)cc2[nH]c1=O"
            }]
            count =0 
            for inputs in sample_inputs:
                count += 1
                result = await app.ainvoke(inputs)
                logger.info("%s", result["messages"][-1].content)
                #logger.debug("Final reports: %s", result["final_reports"])
                final_recipes = []
                # Track which pathways actually succeeded for better reporting
                successful_pathway_ids = []
                failed_pathway_ids = []

                for i in range(len(result["final_reports"])):
                    curr = result["final_reports"][i]
                    curr_pathway_id = result["pathway_id"][i] if i < len(result["pathway_id"]) else "unknown"
                    # 1. Skip if the report is a failure log (the "ERROR" prefix from your map node)
                    if "ERROR for" in curr or "PATHWAY ANALYSIS:" not in curr:
                        failed_pathway_ids.append(curr_pathway_id)
                        continue
                    match = re.search(r'\{.*\}', curr, re.DOTALL)
                    #match = re.search(r'\{.*\}', curr)
                    if match:
                        try:
                            json_str = match.group(0)
                            data = json.loads(json_str)  # Convert string to Python dict
                            #logger.debug("Final reports: %s", result["final_reports"])
                            final_recipes.append(data)
                            successful_pathway_ids.append(curr_pathway_id)
                        except json.JSONDecodeError as e:
                            logger.error("JSON parse failed for pathway %s: %s", curr_pathway_id, e)
                            failed_pathway_ids.append(curr_pathway_id)
                        
                entry = {
                    "idx": count,   #the index among all 500 samples
                    "patent_products": inputs["products_SMILES"],
                    "pathways": result["pathways"],
                    "pathway_id": result["pathway_id"],
                    "successful_pathway_ids": successful_pathway_ids,
                    "failed_pathway_ids": failed_pathway_ids,
                    "retrieval_first": result["top5_retrieval"],
                    "top3_rerank_retrieval":result["top3_rerank_retrieval"],
                    "top3_rerank_extract":result["top3_rerank_extract"], 
                    "full_recipe": final_recipes,
                    "final_reports": result["final_reports"]
                }
                with open(to_save, "a") as f:
                    #f.write(json.dumps(entry) + "\n")
                    json.dump(entry, f, indent=4)
                    f.write("\n")
    
# Entry point to run the async code
if __name__ == "__main__":
    asyncio.run(main())
    #update, add argparse with one simple target molecule SMILES
    


