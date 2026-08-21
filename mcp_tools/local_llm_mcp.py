import torch
from mcp.server.fastmcp import FastMCP
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

# 1. Initialize MCP Server
#mcp = FastMCP("Organic Synthesis Agent")
mcp = FastMCP("SynthesisServer", host="0.0.0.0", port=8000)

#argment parse with local LLM CKPT
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
args = parser.parse_args()

# 2. Global persistent model loading
def setup_model(MODEL_PATH:str):
    #MODEL_PATH = "../../llm/checkpoint-44000"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)#, use_fast=False)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
    )
    return model, tokenizer

model, tokenizer = setup_model(args.model_path)

@mcp.tool()
async def generate_synthesis_results(prompts: list[str], max_tokens: int = 512, num_samples: int = 1, do_sample: bool = True, temperature: float = 0.3, top_p: float = 0.8, top_k: int = 20) -> list[str]:
    """
    Generates Reactants SMILE or reaction conditions include reagents and/or solvents SMILES strings using the local SFT model based on query condition.
    Use this for high-precision organic chemistry tasks.
    """
    #inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    bs =len(prompts)
    messages = []
    for j in range(bs):
        native = [{"role":"user", "content": prompts[j]}]
        messages.append(native)
    text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True, # Must add for generation
            enable_thinking=False,
        )#.to(device)
    inputs = tokenizer(text, padding=True,truncation=True,return_tensors = "pt").to(model.device)
    seq_len = len(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model.generate(
                **inputs,
                # Recommended Qwen3 settings for non-reasoning
                do_sample = False, #do_sample, #for multiple returns #num_return_sequences=num_samples, # top_5 generations
                num_return_sequences=num_samples,
                use_cache=True,
                #for top1
                temperature = 0.0,
                #temperature = temperature, top_p = top_p , top_k = top_k,
                num_beams=num_samples,
                # for top5,
                #temperature = 0.9, top_p = 0.95, top_k = 65, #for first test, temperature 0.7, top_p 0.8, top_k 20 for greedy1
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                )
    
    decoded = tokenizer.batch_decode(outputs[:, seq_len:], skip_special_tokens=True)
    return decoded

if __name__ == "__main__":
    # Expose as a web service on port 8000
    mcp.run(transport="streamable-http")

