from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core import rate_limiters #import InMemoryRateLimiter #add rate limiter
import os

try:
    from dotenv import load_dotenv

    _PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_PKG_ROOT, ".env"))
except ImportError:
    pass

import logging
##Basic log configuration: INFO level, includes timestamp, module name, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# add rate_limiter, TODO
# Allow 1 request every 2 seconds (0.5 requests per second)
rate_limiter = rate_limiters.InMemoryRateLimiter(
    requests_per_second=0.1,
    check_every_n_seconds=1,
    max_bucket_size=10
)

sota_model = init_chat_model(
    model=os.getenv("SOTA_MODEL", "gemini-2.5-pro"),#"gemini-3.1-pro-preview", #opus_4_5, #"gemini-2.5-pro", # Specify any OpenRouter model ID
    model_provider="openai",             # OpenRouter uses the OpenAI-compatible provider
    api_key=os.getenv("SOTA_API_KEY"),
    base_url=os.getenv("SOTA_BASE_URL", "https://www.litellm.org/"),
    rate_limiter=rate_limiter,
    temperature=float(os.getenv("SOTA_TEMPERATURE", "1.0")),
    streaming=True
)


prompt_org = ChatPromptTemplate.from_template("""
        You are a chemistry expert. Analyze this specific pathway:
        Reactants: {reactants} | Product: {product}
        Conditions: {conditions}
        Analyze the whole reaction first. Provide a detailed ranking and recommendation for this specific pathway in short way. If this whole recipe is not plausible or not optimal include initial rectants, optimize the single_step retro_synthesis recipe or propose your own single-step retro_synthesis pathway. Finally give recommendation in JSON format include: 1. reaction_name, 2. recommendation_and_ranking (with ranking, rational, confidence score(1-5, based on whole recipes).).        3. All organic synthesis reaction_conditions inculde reactants, reagents, solvents, temperature, quantity recommendation of every component (specific number of temperature in unit of °C, specific number of reaction time in units of hours, quantity recommendation with equivalents, every component include name, smiles, volumes),
        4. procedure.
        Final JSON with key words include: reaction_name, recommendation_and_ranking, reaction_conditions(inner keys include reactants, reagents, solvents, temperature, time), procedure.
        """)

prompt_new_march = ChatPromptTemplate.from_template("""
You are an expert synthetic organic chemist. Your task is to critically analyze and optimize a proposed single-step retrosynthetic pathway.

INPUT DATA:
- Proposed Reactants: {reactants}
- Target Product: {product}
- Proposed Conditions: {conditions}

Follow INSTRUCTIONS below to reason step-by-step first:
1. Pathway Analysis: Evaluate the plausibility, atom economy, and overall viability of the proposed pathway. 
2. Optimization & Fallback: If the proposed recipe is plausible, optimize the reaction conditions. If the proposed recipe is NOT plausible or highly suboptimal, propose a superior alternative single-step retrosynthetic pathway to reach the target product.
3. Concise Evaluation: Provide a concise but comprehensive rationale and a confidence score (1-5) based on the final recommended recipe.
4. Chemical Specificity: All components must include common names, SMILES strings, and stoichiometric equivalents (or volume equivalents for solvents). Temperatures must be in strictly numeric °C, and time in strictly numeric hours.
5. Formatting: You must output ONLY a valid JSON object. Do not include introductory text, markdown formatting blocks (like ```json), or concluding remarks.

Finally use the exact JSON schema below:

{{
  "reaction_name": "String (e.g., 'Suzuki-Miyaura Cross-Coupling')",
  "recommendation_and_ranking": {{
    "ranking": "String (e.g., 'Excellent', 'Good', 'Fair', 'Poor')",
    "rationale": "String (Concise explanation of why this pathway was chosen or modified)",
    "confidence_score": "Integer (1-5)"
  }},
  "reaction_conditions": {{
    "reactants": [
      {{"name": "String", "smiles": "String", "equivalents": "Float"}}
    ],
    "reagents": [
      {{"name": "String", "smiles": "String", "equivalents": "Float"}}
    ],
    "solvents": [
      {{"name": "String", "smiles": "String", "volume_ratio": "String (e.g., '10 mL/mmol')"}}
    ],
    "temperature": "Integer (Degrees Celsius)",
    "time": "Float (Hours)"
  }},
  "procedure": [
    "String (Step 1)",
    "String (Step 2)"
  ]
}}""")

async def pathway_generate_only_node(state: dict, mcp_retrieve):
        """
        Parallel Branch: Executed simultaneously for every reactant.
        """
        # 3. Individual Generation (Your specific prompt logic)
        prompt = ChatPromptTemplate.from_template("""
        You are a chemistry expert. Analyze this specific pathway:
        Reactants: {reactants} | Product: {product}
        Conditions: {conditions}
        
        Analyze the whole reaction first. Provide a detailed ranking and recommendation for this specific pathway in short way. If this whole recipe is not plausible or not optimal include initial rectants, optimize the single_step retro_synthesis recipe or propose your own single-step retro_synthesis pathway. Finally give recommendation in JSON format include: 1. reaction_name, 2. recommendation_and_ranking (with ranking, rational, confidence score(1-5, based on whole recipes).).
        3. All organic synthesis reaction_conditions inculde reactants, reagents, solvents, temperature, quantity recommendation of every component (number of temperature in unit of °C, number of reaction time in units of hours, quantity recommendation with equivalents, solvents' volume in unit of ML, every component include name, smiles, volumes),
        4. procedure.
        Final JSON with key words include: reaction_name, recommendation_and_ranking, reaction_conditions(inner keys include reactants (with keys: name, smiles, equivalents), reagents (with keys: name, smiles, equivalents), solvents (with keys: name, smiles, volume), temperature, time), procedure.
        """)
        
        chain = prompt | sota_model #sota_model
        response = await chain.ainvoke({
            "reactants": state["reactants"],
            "product": state["products_SMILES"],
            "conditions": state["condition"],
        })

        # 3. Return to the aggregator
        #logger.debug("SHAN: %s", response)
        return {"final_reports": [f"PATHWAY ANALYSIS:\n{response.content}"]}
