from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
import logging
from nodes.rag_node import invoke_and_validate

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
    requests_per_second=0.1,
    check_every_n_seconds=1,
    max_bucket_size=10
)
#strict
rate_limiter_strict = rate_limiters.InMemoryRateLimiter(
    requests_per_second=0.04,
    check_every_n_seconds=0.1,
    max_bucket_size=1
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
    model=os.getenv("SOTA_MODEL", "gemini-2.5-pro"), #"gemini-3.1-pro-preview", #opus_4_5, #"gemini-2.5-pro", # Specify any OpenRouter model ID
    model_provider="openai",             # OpenRouter uses the OpenAI-compatible provider
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


async def pathway_simple_pipeline_node(state: dict, mcp_retrieve):
        """
        Parallel Branch: Executed simultaneously for every reactant.
        TO use GNN rank only
        """
        logger.debug("===debug current pathway pipeline is called!!! %s", state)
        # 1. Individual RAG (FAISS) call for this specific reactant
        # We call your MCP RAG tool directly here
        docs = await mcp_retrieve.ainvoke({
            "reactants_smiles": state["reactants"],
            "products_smiles": state["products_SMILES"],
            "k":3
        })
        logger.info("✅ Retrieval part DONE!")
        logger.info(f"doc is {docs[0]} and type of docs[0] is {type(docs[0])} and docs[0]['results'] is {docs[0]['text']}")
        #choose top3 samples only
        data = docs[0]['text']

        
        # 3. Individual Generation (Your specific prompt logic)
        prompt = ChatPromptTemplate.from_template("""
        You are a chemistry expert. Analyze this specific pathway:
        Reactants: {reactants} | Product: {product}
        Conditions: {conditions}
        
        Similar reaction for reaction conditions recommendation retrieval from FAISS:
        {context}
        
        Analyze the whole reaction first. Provide a detailed ranking and recommendation for this specific pathway in short way. If this whole recipe is not plausible or not optimal include initial rectants, optimize the single_step retro_synthesis recipe or propose your own single-step retro_synthesis pathway. Finally give recommendation in JSON format include: 1. reaction_name, 2. recommendation_and_ranking (with ranking, rational, confidence score(1-5, based on whole recipes).).
        3. All organic synthesis reaction_conditions inculde reactants, reagents, solvents, temperature, volumes recommendation of every component (specific number of temperature in unit of °C, specific number of reaction time in units of hours, quantity recommendation with equivalents, solvents' volume in unit of ML, every component include name, smiles, volumes),
        4. procedure.
        Final JSON with key words include: reaction_name, recommendation_and_ranking, reaction_conditions(inner keys include reactants (with keys: name, smiles, equivalents), reagents (with keys: name, smiles, equivalents), solvents (with keys: name, smiles, volume), temperature, time), procedure.
        """)
        
        chain = prompt | sota_model #sota_model
        
        response = await invoke_and_validate(chain, state, data=data)
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
                "top5_retrieval": [f"{docs}"],
                "pathway_id": [state["pathway_id"]],
                "top3_rerank_extract": [f"{data}"],
                "top3_rerank_retrieval": [f"{data}"]}
