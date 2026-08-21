from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
import re
import logging
import json
import ast
import asyncio
from rdkit import Chem
from utils.reaction_plausibility import (
    check_reactant_scaffold_conservation,
    is_reaction_element_consistent,
)
from langchain_core.runnables import RunnableConfig
from langchain_core import rate_limiters
import os

try:
    from dotenv import load_dotenv

    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_PKG_ROOT, ".env"))
except ImportError:
    pass

# add rate_limiter, TODO
# Allow 1 request every 2 seconds (0.5 requests per second)
rate_limiter = rate_limiters.InMemoryRateLimiter(
    requests_per_second=1,
    check_every_n_seconds=1,
    max_bucket_size=10
)
#strict
rate_limiter_strict = rate_limiters.InMemoryRateLimiter(
    requests_per_second=1,
    check_every_n_seconds=1,
    max_bucket_size=10
)

##Basic log configuration: INFO level, includes timestamp, module name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#version 2 sota model sonnet-4-5 or opus-4-5
sonnet_4_5 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
opus_4_5 = "us.anthropic.claude-opus-4-5-20251101-v1:0"
sota_model = init_chat_model(
    model=os.getenv("SOTA_MODEL", "gemini-2.5-pro"),  # Specify any OpenRouter model ID
    model_provider="openai",              # OpenRouter uses the OpenAI-compatible provider
    api_key=os.getenv("SOTA_API_KEY"),
    base_url=os.getenv("SOTA_BASE_URL", "https://www.litellm.org/"),
    rate_limiter=rate_limiter_strict,
    temperature=float(os.getenv("SOTA_TEMPERATURE", "1.0")),
    streaming=True
)
#after feb2026
rerank_llm = init_chat_model(
    model=os.getenv("RERANK_MODEL", "gemini-2.5-flash"), # Specify any OpenRouter model ID
    model_provider="openai",             # OpenRouter uses the OpenAI-compatible provider
    api_key=os.getenv("RERANK_API_KEY"),   
    base_url=os.getenv("RERANK_BASE_URL", "https://www.litellm.org/"),
    rate_limiter=rate_limiter,
    temperature=float(os.getenv("RERANK_TEMPERATURE", "1.0")),
    streaming=True
)

class RDKitProcessingError(Exception):
    """Raised when RDKit fails to process a molecule."""
    def __init__(self, smiles, stage, details="", payload=None):
        self.smiles = smiles
        self.stage = stage
        self.details = details
        self.payload = payload
        super().__init__(f"Error during {stage} for: {smiles}. {details}")


def extract_json_molecules(text, target_molecule, scaffold_validation=True):
    """
    check whether or not to pass the RDKit
    """
    pred_reactant_smiles_list = []
    try:
        pred_reactant_smiles_list = [
            item["smiles"] for item in text["reaction_conditions"]["reactants"]
        ]
        pred_reactants = '.'.join(pred_reactant_smiles_list)
    except Exception as e:
        RDKitProcessingError(text["reaction_conditions"]["reactants"], "Parsing", "Invalid SMILES syntax: " + str(e))
        pred_reactants = None
    try:
        pred_reagents = '.'.join([item["smiles"] for item in text["reaction_conditions"]["reagents"]])
    except Exception as e:
        RDKitProcessingError(text["reaction_conditions"]["reagents"], "Parsing", "Invalid SMILES syntax " + str(e))
        pred_reagents = None
    try:
        pred_solvents = '.'.join([item["smiles"] for item in text["reaction_conditions"]["solvents"]])
    except Exception as e:
        RDKitProcessingError(text["reaction_conditions"]["solvents"], "Parsing", "Invalid SMILES syntax: " + str(e))
        pred_solvents = None
    try:
        pred_reactants = Chem.MolToSmiles(Chem.MolFromSmiles(pred_reactants))
    except Exception as e:
        RDKitProcessingError(pred_reactants, "Canonical", "Invalid SMILES reactants Canonical error: " + str(e))
        pred_reactants = None
    try:
        pred_reagents = Chem.MolToSmiles(Chem.MolFromSmiles(pred_reagents))
    except Exception as e:
        RDKitProcessingError(pred_reagents, "Canonical", "Invalid SMILES reagents Canonical error: " + str(e))
        pred_reagents = None
    try:
        pred_solvents = Chem.MolToSmiles(Chem.MolFromSmiles(pred_solvents))
    except Exception as e:
        RDKitProcessingError(pred_solvents, "Canonical", "Invalid SMILES solvents Canonical error: " + str(e))
        pred_solvents = None
    if pred_solvents is None:
        return False, "Invalid or missing solvent SMILES"
        
    try:
        valid = is_reaction_element_consistent([pred_reactants, pred_reagents],target_molecule)
    except Exception as e:
        return False, f"Element consistency check crashed: {e}"
    if not valid:
        return False, "Product contains elements not present in reactants/reagents"

    if scaffold_validation:
        try:
            scaffold_check = check_reactant_scaffold_conservation(
                pred_reactant_smiles_list,
                target_molecule,
            )
            if not scaffold_check["valid"]:
                logger.warning(
                    "Reactant scaffold validation failed: %s",
                    scaffold_check["message"],
                )
                return False, f"Scaffold validation failed: {scaffold_check['message']}"
        except Exception as e:
            logger.warning("Reactant scaffold validation crashed: %s", e)
            return False, f"Scaffold validation crashed: {e}"
    return True, ""

def _split_reactant_components(reactants_smiles: str):
    return [
        part.strip()
        for part in str(reactants_smiles or "").split(".")
        if part.strip()
    ]


def _build_local_scaffold_validation_block(state: dict) -> str:
    if not state.get("scaffold_validation", True):
        return (
            "Local reactant scaffold pre-check: DISABLED.\n"
            "Because no scaffold pre-check is available, preserve the listed Reactants "
            "by default and revise them only with explicit, strong chemical evidence.\n"
        )

    reactants = state.get("reactants", "")
    product = state.get("products_SMILES", "")
    try:
        scaffold_check = check_reactant_scaffold_conservation(
            _split_reactant_components(reactants),
            product,
        )
    except Exception as e:
        logger.warning("Local reactant scaffold pre-check crashed: %s", e)
        return (
            "Local reactant scaffold pre-check: ERROR.\n"
            f"Error: {e}\n"
            "Preserve the listed Reactants by default. Revise reactant SMILES only if "
            "there is explicit, strong chemical evidence that the local reactants cannot "
            "form the target product.\n"
        )

    if scaffold_check.get("valid"):
        return (
            "Local reactant scaffold pre-check: PASSED.\n"
            "The local model's Reactants conserve the target product scaffold. Therefore, "
            "preserve the original Reactants SMILES exactly in reaction_conditions.reactants "
            "and set reactant_revision.changed=false by default. Do not change component "
            "membership, substituent positions, stereochemistry, salts, or SMILES connectivity "
            "just to make the route look cleaner. You may revise reactant SMILES only if you "
            "can provide very strong evidence from the product connectivity and retrieved "
            "reactions that the local Reactants are impossible. If such a rare revision is "
            "made, reactant_revision.reason and evidence_from_retrieval must explicitly state "
            "why the scaffold-passing local Reactants are still impossible.\n"
        )

    return (
        "Local reactant scaffold pre-check: FAILED.\n"
        f"Validation error: {scaffold_check.get('message', 'unknown scaffold mismatch')}\n"
        "The local model's Reactants do not reliably conserve the target product scaffold. "
        "In this case you are allowed to revise reactant SMILES, but the revised reactants "
        "must form the exact target product and conserve non-reacting scaffolds, substituent "
        "positions, stereochemistry, and atom connectivity. Explain the correction in "
        "reactant_revision.\n"
    )


def _local_scaffold_status(inputs: dict) -> str:
    block = inputs.get("local_scaffold_validation_block", "") or ""
    if "Local reactant scaffold pre-check: FAILED" in block:
        return "FAILED"
    if "Local reactant scaffold pre-check: PASSED" in block:
        return "PASSED"
    if "Local reactant scaffold pre-check: DISABLED" in block:
        return "DISABLED"
    if "Local reactant scaffold pre-check: ERROR" in block:
        return "ERROR"
    return "UNKNOWN"


def _reactant_entries_from_local(local_reactants: str, existing_reactants=None):
    existing_reactants = existing_reactants if isinstance(existing_reactants, list) else []
    entries = []
    for idx, smiles in enumerate(_split_reactant_components(local_reactants)):
        old = existing_reactants[idx] if idx < len(existing_reactants) and isinstance(existing_reactants[idx], dict) else {}
        entries.append({
            "name": old.get("name") or f"local_reactant_{idx + 1}",
            "smiles": smiles,
            "equivalents": old.get("equivalents", 1.0),
        })
    return entries


def _enforce_reactant_revision_policy(parsed_data: dict, inputs: dict):
    if not isinstance(parsed_data, dict) or "reactants" not in inputs:
        return False

    reaction_conditions = parsed_data.setdefault("reaction_conditions", {})
    if not isinstance(reaction_conditions, dict):
        return False

    current_reactants = reaction_conditions.get("reactants", [])
    reactant_revision = parsed_data.get("reactant_revision")
    if not isinstance(reactant_revision, dict):
        reactant_revision = {}
        parsed_data["reactant_revision"] = reactant_revision

    status = _local_scaffold_status(inputs)
    changed = reactant_revision.get("changed")
    # Reactant SMILES may be overwritten only when the local scaffold pre-check
    # explicitly failed. In all other states, preserve local model reactants.
    must_preserve_local = status != "FAILED" or changed is False
    if not must_preserve_local:
        return False

    if current_reactants:
        reactant_revision.setdefault("alternative_reactants", current_reactants)
    reactant_revision["changed"] = False
    reactant_revision["original_reactants"] = inputs.get("reactants", "")
    reactant_revision["final_reactants"] = inputs.get("reactants", "")
    reactant_revision.setdefault(
        "policy_note",
        (
            "reaction_conditions.reactants were forced to the local model reactants "
            "because local scaffold pre-check did not fail or reactant_revision.changed "
            "was false."
        ),
    )
    reaction_conditions["reactants"] = _reactant_entries_from_local(
        inputs.get("reactants", ""),
        current_reactants,
    )
    return True


def _replace_json_block(content: str, match, parsed_data: dict) -> str:
    fixed_json = json.dumps(parsed_data, ensure_ascii=False, indent=2)
    return content[:match.start(1)] + fixed_json + content[match.end(1):]


def _build_validation_feedback(error: Exception, attempt_number: int) -> str:
    if isinstance(error, RDKitProcessingError):
        if "Scaffold validation failed" in error.details:
            return (
                f"\nPrevious attempt {attempt_number} failed scaffold validation.\n"
                f"Validation error: {error.details}\n"
                "For the next attempt, revise the reactant SMILES so that every "
                "non-reacting local scaffold and substituent position in each "
                "substantial reactant is conserved in the product. Do not change "
                "reactants only to copy retrieved examples; use retrieval mainly "
                "for conditions unless the proposed reactants are chemically invalid.\n"
            )
        return (
            f"\nPrevious attempt {attempt_number} failed output validation.\n"
            f"Validation error: {error.details}\n"
            "For the next attempt, fix the JSON and reaction_conditions fields while "
            "keeping the proposed single-step retrosynthesis chemically plausible.\n"
        )
    return (
        f"\nPrevious attempt {attempt_number} failed due to an execution error: {error}\n"
        "For the next attempt, return a complete valid JSON recipe.\n"
    )


async def invoke_and_validate(chain, inputs:dict, config: RunnableConfig, data=None, node_type="RAG", history=None):
    validation_feedback_block = ""
    max_attempts = 5
    last_error = None

    for attempt_number in range(1, max_attempts + 1):
        try:
            # 1. Execute LLM
            # generate node type, with retrieval RAG or native fallback
            if node_type == "RAG":
                response = await chain.ainvoke({
                        "reactants": inputs["reactants"],
                        "product": inputs["products_SMILES"],
                        "conditions": inputs["condition"],
                        "context": data, #ranked_docs.content #str(docs)
                        "local_scaffold_validation_block": inputs.get("local_scaffold_validation_block", ""),
                        "validation_feedback_block": validation_feedback_block,
                    },config=config)
            else:
                response = await chain.ainvoke({
                    "product": inputs["products_SMILES"],
                    "validation_feedback_block": validation_feedback_block,
                },config=config)
            #response = await chain.ainvoke(inputs)
            
            # --- STAGE 1: JSON & Keyword Extraction ---
            # Assuming response is a string or has a 'text' attribute
            content = response if isinstance(response, str) else response.content
            if history is not None:
                history.append(content)
            pattern = r"```json\s*(\{.*?\})\s*```"
            match = re.search(pattern, content, re.DOTALL)
            parsed_data = None
            if not match:
                raise RDKitProcessingError(inputs["products_SMILES"], "Extraction", "No JSON found", payload=response)

            json_str = match.group(1).strip()
            try:
                # Step 1: Try standard JSON (strict)
                parsed_data = json.loads(json_str)
            except Exception:
                try:
                    parsed_data = ast.literal_eval(json_str)
                except Exception as e:
                    raise RDKitProcessingError(inputs["products_SMILES"], "Pasing", f"Failed to parse JONS after repair: {str(e)}", payload=response)
                #final_recipes.append(data)    
            if node_type == "RAG":
                policy_changed = _enforce_reactant_revision_policy(parsed_data, inputs)
                if policy_changed:
                    content = _replace_json_block(content, match, parsed_data)
                    if isinstance(response, str):
                        response = content
                    else:
                        response.content = content
            # Check for required keywords/keys
            valid, reason = extract_json_molecules(
                parsed_data,
                inputs["products_SMILES"],
                scaffold_validation=inputs.get("scaffold_validation", True),
            )
            
            if not valid:
                    raise RDKitProcessingError(
                inputs["products_SMILES"],
                "Validation",
                reason,  
                payload=response
                )

            return response
        except Exception as e:
            last_error = e
            if attempt_number >= max_attempts:
                raise
            validation_feedback_block = (
                "\nFeedback from previous failed validation attempts:\n"
                f"{_build_validation_feedback(e, attempt_number)}"
            )
            logger.info(
                "Validation attempt %s/%s failed for product %s: %s",
                attempt_number,
                max_attempts,
                inputs["products_SMILES"],
                getattr(e, "details", str(e)),
            )
            await asyncio.sleep(2)

    raise last_error



async def pathway_pipeline_node(state: dict, config: RunnableConfig,mcp_retrieve):
        """
        Parallel Branch: Executed simultaneously for every reactant.
        """
        logger.debug("===debug current pathway pipeline is called!!! %s", state)
        # 1. Individual RAG (FAISS) call for this specific reactant
        # We call your MCP RAG tool directly here
        docs = await mcp_retrieve.ainvoke({
            "reactants_smiles": state["reactants"],
            "products_smiles": state["products_SMILES"]
        })
        logger.info("✅ Retrieval part DONE!")
        
        # 2. Rerank Step (Using a separate, superior LLM)
        # ---------------------------------------------------------
        rerank_prompt = ChatPromptTemplate.from_template("""
        You are a organic chemistry reranking expert. 
        Target Reaction: {reactants} -> {products}
        
        Candidates from FAISS:
        {docs}
        
        TASK: ALWAYS first analyse the target reaction and identify key disconnections. Think step by step. Rerank all retrieval similar reaction. Select the TOP 3 most chemically relevant reactions. 
        Output FINALLY the raw content of those 3 reactions in ONE JSON format with ```json\n [ .....]\n```.
        """)
        
        # Use your 'superior' model here (e.g., GPT-4o or a large Qwen)
        rerank_config = config.copy()
        rerank_config["tags"] = rerank_config.get("tags", []) + ["rag_chain", "rerank"]
        rerank_chain = rerank_prompt | rerank_llm 
        ranked_docs = await rerank_chain.ainvoke({
            "reactants": state["reactants"],
            "products": state["products_SMILES"],
            "docs": str(docs)
        },config=rerank_config)
        
        logger.info("✅ Rerank Complete. Context pruned for final generation.")
        #pattern = r"json\s*(\[.*?\])\s*"
        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, ranked_docs.content, flags=re.DOTALL)
        data = None
        if match:
            json_str = match.group(1)
            data = json_str #json.loads(json_str)
        logger.debug("Current ranked documents are %s", data)
        local_scaffold_validation_block = _build_local_scaffold_validation_block(state)
        citations = []
        for m in re.finditer(r'(?:\\*)"file_name(?:\\*)"\s*:\s*(?:\\*)"([^"\\]+)(?:\\*)"', ranked_docs.content or "", re.I):
            fn = m.group(1).strip().replace("\\", "")
            if re.fullmatch(r"US\d+", fn):
                citations.append({"file_name": fn, "rank": len(citations) + 1})
            if len(citations) >= 3:
                break
        
        # 3. Individual Generation (Your specific prompt logic)
        prompt = ChatPromptTemplate.from_template("""
        You are a chemistry expert. Analyze this specific pathway:
        Reactants: {reactants} | Product: {product}
        Conditions: {conditions}
        
        Similar reaction for reaction conditions recommendation retrieval from FAISS:
        {context}

        {local_scaffold_validation_block}

        {validation_feedback_block}
        
        Analyze the whole reaction first. We aim to synthesize the target product. The listed Reactants are the local model's proposed single-step retrosynthesis and should be preserved by default.
        Follow the local reactant scaffold pre-check above as the primary policy for whether reactant SMILES may be revised. If the pre-check PASSED, keep reaction_conditions.reactants identical to the listed Reactants except for harmless ordering. In that case reactant_revision.changed should be false. Only override this when there is very strong, explicit evidence that the scaffold-passing local Reactants are still impossible.
        If the pre-check FAILED, you may revise the reactants to repair the scaffold/connectivity problem. The new reactants must plausibly form the exact target product and must conserve non-reacting local scaffolds, substituent positions, stereochemistry, and atom connectivity in the product.
        Do not replace reactants merely because a retrieved example uses different substrates; retrieved reactions are primarily evidence for reaction type and conditions. If you keep the original reactants, still verify and explain why they are plausible.
        Provide a detailed ranking and recommendation for this specific pathway in a short way. Finally give recommendation in JSON format include: 1. reaction_name, 2. recommendation_and_ranking (with ranking, rational, confidence score(1-5, based on whole recipes).),
        3. reactant_revision (inner keys include changed, original_reactants, final_reactants, reason, evidence_from_retrieval, scaffold_conservation_rationale),
        4. All organic synthesis reaction_conditions inculde reactants, reagents, solvents, temperature, volumes recommendation of every component (specific number of temperature in unit of °C, specific number of reaction time in units of hours, quantity recommendation with equivalents, solvents' volume in unit of ML, every component include name, smiles, volumes),
        5. procedure.
        Final JSON with key words include: reaction_name, recommendation_and_ranking, reactant_revision, reaction_conditions(inner keys include reactants (with keys: name, smiles, equivalents), reagents (with keys: name, smiles, equivalents), solvents (with keys: name, smiles, volume), temperature, time), procedure.
        """)
        
        claude_prompt = ChatPromptTemplate.from_template("""
        You are a chemistry expert. Analyze this specific pathway:
        Reactants: {reactants} | Product: {product}
        Conditions: {conditions}

        Similar reaction for reaction conditions recommendation retrieval from FAISS:
        {context}

        Analyze the target molecules and the plausible of the reactants to synthesize the target molecules. Analyze the whole reaction then. If reactants is not plausible or conditions not plausible, optimize the single_step retro_synthesis recipe or propose your own reaction chemical identities for single-step retro_synthesis. Provide a detailed ranking and recommendation for this specific pathway in short way. If this whole recipe is not optimal include initial rectants, optimize the single_step retro_synthesis recipe. 
        Finally give recommendation in JSON format include: 
        1. reaction_name, 2. recommendation_and_ranking (with ranking, target_molecule_property, plausiblity_of_reactants, rational_of_whole_reaction, confidence score(1-5, based on whole recipes).).
        3. All organic synthesis reaction_conditions inculde reactants, reagents, solvents, temperature, volumes recommendation of every component (specific number of temperature in unit of °C, specific number of reaction time in units of hours, quantity recommendation with equivalents, solvents' volume in unit of ML, every component include name, smiles, volumes),
        4. procedure.
        Final JSON with key words include: reaction_name, recommendation_and_ranking, reaction_conditions(inner keys include reactants (with keys: name, smiles, equivalents), reagents (with keys: name, smiles, equivalents), solvents (with keys: name, smiles, volume), temperature, time), procedure.
        """)
        chain = prompt | sota_model #sota_model
        # chain = claude_prompt | sota_model #sota_model

        gen_config = config.copy()
        gen_config["tags"] = gen_config.get("tags", []) + ["rag_chain","generate"]
        generation_history = []
        generation_state = dict(state)
        generation_state["local_scaffold_validation_block"] = local_scaffold_validation_block
        try:
            response = await invoke_and_validate(chain, generation_state, data=data, config=gen_config, history=generation_history)
            """
            response = await chain.ainvoke({
                "reactants": state["reactants"],
                "product": state["products_SMILES"],
                "conditions": state["condition"],
                "context": data #ranked_docs.content #str(docs)
            })"""

            # 3. Return to the aggregator
            #logger.info("SHAN: %s", state["pathway_id"])
            return {"final_reports": [f"PATHWAY ANALYSIS:\n{response.content}"],
                    "validation_errors": [{
                    "pathway_id": state["pathway_id"],
                    "stage": "complete",
                    "details": "",
                    "type": "success",
                    }],
                    "top5_retrieval": [f"{docs}"],
                    "pathway_id": [state["pathway_id"]],
                    "top3_rerank_extract": [f"{data}"],
                    "top3_rerank_retrieval": [f"{ranked_docs.content}"],
                    "history": [generation_history],
                    "citations": [citations]}
        except Exception as e:
            if isinstance(e, RDKitProcessingError) and "Scaffold" in e.details and getattr(e, "payload", None) is not None:
                payload_content = e.payload if isinstance(e.payload, str) else getattr(e.payload, "content", str(e.payload))
                return {
                    "final_reports": [f"PATHWAY ANALYSIS (SCAFFOLD VALIDATION WARNING: {e.details}):\n{payload_content}"],
                    "validation_errors": [{
                        "pathway_id": state["pathway_id"],
                        "stage": e.stage,
                        "details": e.details,
                        "type": "scaffold",
                    }],
                    "top5_retrieval": [f"{docs}"],
                    "pathway_id": [state["pathway_id"]],
                    "top3_rerank_extract": [f"{data}"],
                    "top3_rerank_retrieval": [f"{ranked_docs.content}"],
                    "history": [generation_history],
                    "citations": [citations]
                }
            
            # Fallback for other errors (e.g., Parsing or Extraction failures)
            return {"final_reports": [f"ERROR for {state['pathway_id']}: {str(e)}"],
                    "validation_errors": [{
                        "pathway_id": state["pathway_id"],
                        "stage": getattr(e, "stage", "unknown"),
                        "details": getattr(e, "details", str(e)),
                        "type": "error",
                    }],
                    "top5_retrieval": [None],
                    "pathway_id": [state["pathway_id"]],
                    "top3_rerank_extract": [None],
                    "top3_rerank_retrieval": [None],
                    "history": [generation_history],
                    "citations": [citations]}
        

async def fallback_agent_node(state: dict, config: RunnableConfig):
        """
        fallback agent for empty effetive reactants recommendation case. SOTA LLM perform this final directly based on given products.
        """
        prompt = ChatPromptTemplate.from_template("""
        You are a chemistry expert. Analyze/Reasoning this Product(s) with given SMILES: {product}. Identify effective bonds of this organic synthesis Propose plausible and effective reactants, reagents, solvents.

        {validation_feedback_block}
        
        Analyze the whole reaction first. Propose your own single-step retro_synthesis pathway. Finally give recommendation in JSON format include: 1. reaction_name, 2. recommendation_and_ranking (rational, confidence score(1-5, based on whole recipes).).
        3. All organic synthesis reaction_conditions inculde reactants, reagents, solvents, temperature, quantitiy recommendation of every component (specific number of temperature in unit of °C, specific number of reaction time in units of hours, quantity recommendation with equivalents, solvents' volumne in unit of ML),
        4. procedure.
        Final JSON with key words include: reaction_name, recommendation_and_ranking, reaction_conditions(inner keys include reactants (with keys: name, smiles, equivalents), reagents (with keys: name, smiles, equivalents), solvents (with keys: name, smiles, volume), temperature, time), procedure.
        """)

        chain = prompt | sota_model #sota_model
        gen_config = config.copy()
        gen_config["tags"] = gen_config.get("tags", []) + ["fallback","generate"]
        response = await invoke_and_validate(chain, state, config=gen_config, node_type="native")

        # 3. Return to the aggregator
        #logger.debug("SHAN: %s", response)
        return {"final_reports": [f"PATHWAY ANALYSIS:\n{response.content}"],
                "top5_retrieval": [None],
                "top3_rerank_extract": [None],
                "top3_rerank_retrieval": [None],
                "citations": [None]}
