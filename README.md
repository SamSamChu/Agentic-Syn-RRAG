# Agentic SynR-RAG

Agentic SynR-RAG is a retrieval-augmented framework for single-step retrosynthesis and executable synthesis-protocol generation. It combines a local SFT pathway model, reaction-graph retrieval, semantic reranking, and an external reasoning model in a Generate–Retrieve–Refine workflow.

> This repository contains research code. Generated routes and procedures must be reviewed by qualified chemists before experimental use.

## Overview

Given a target product SMILES, SynR-RAG performs four stages:

1. **Pathway generation** — a local SFT model proposes reactants, reagents, and solvents.
2. **Structural retrieval** — RXNGraphormer embeddings and a FAISS index retrieve related patent reactions.
3. **Semantic reranking** — an LLM reranks retrieved precedents by chemical and procedural relevance.
4. **Protocol refinement** — a frontier model revises the pathway when warranted and produces a structured synthesis recipe.

The repository supports the full SynR-RAG pipeline, a retrieval-free generation ablation, a direct native-LLM baseline, repeated target-molecule tests, and rule-based and LLM-based evaluation.

<p align="center">
  <img src="./images/SynR_RAG.jpg" alt="SynR-RAG architecture" width="720">
</p>

## Key features

- End-to-end Generate–Retrieve–Refine synthesis planning
- Hybrid structural retrieval and semantic reranking
- Top-k pathway generation with canonicalization, verification, deduplication, and reranking
- Batch inference with checkpoints and deterministic index-based sharding
- Full RAG, simple-RAG, retrieval-free, and native-LLM comparison modes
- Reactant accuracy, molecular validity, elemental consistency, structural compatibility, LLM-as-a-judge, and faithfulness evaluation

## Repository layout

```text
.
├── run_app.py                         # Authoritative benchmark/batch runner
├── predict_target_mols.py             # Product-only CSV inference
├── OSCAR_main.py                      # LangGraph construction
├── OSCAR_generate_only_native.py      # Direct native-LLM baseline
├── nodes/                              # Generation, retrieval, reranking, refinement nodes
├── mcp_tools/                          # Local SFT service and reaction-retrieval tools
├── benchmark/                          # Rule-based and LLM-based evaluation scripts
├── examples/                           # Curated, publication-facing result examples
├── pretrained_classification_model/   # RXNGraphormer model assets
├── data/                               # Reaction database and FAISS assets; prepared separately
├── requirements_main.txt              # Main pipeline environment
└── requirements_server.txt            # Local SFT service environment
```

## Installation

The main pipeline and local SFT service use separate environments.

### 1. Clone the repository

```bash
git clone https://github.com/SamSamChu/Agentic-SynR-RAG.git
cd Agentic-SynR-RAG
```

### 2. Main pipeline environment

Python 3.10 is recommended.

```bash
conda create -n agentic-synr-rag python=3.10 -y
conda activate agentic-synr-rag

pip install -r requirements_main.txt \
  -f https://data.pyg.org/whl/torch-2.2.0+cpu.html \
  --extra-index-url https://download.pytorch.org/whl/cpu

git clone -b pytorch2 https://github.com/licheng-xu-echo/RXNGraphormer.git
pip install ./RXNGraphormer
```

### 3. Local SFT service environment

Python 3.12 is recommended for the local generation service.

```bash
conda create -n synr-sft-server python=3.12 -y
conda activate synr-sft-server
pip install -r requirements_server.txt
```

## Required assets

The full RXNGraphormer weights, database, index, benchmark CSV, and local SFT checkpoint are not bundled with this source tree. Arrange the runtime assets as follows:

```text
Agentic-SynR-RAG/
├── pretrained_classification_model/...
├── data/
│   ├── offline_reaction_database.json
│   ├── reaction_update.faiss
│   └── indices_map_update.npy
├── checkpoints/
│   └── <local-sft-checkpoint>/...
└── evaluate/
    └── agent_benchmark_162_clean.csv
```

The RXNGraphormer pretrained classification model is available from [Figshare](https://doi.org/10.6084/m9.figshare.28356077). Extract it into `pretrained_classification_model/`. Use a database, FAISS index, and index map produced as one matched set; an index generated from a different database ordering is invalid.

The benchmark runner expects these CSV columns:

| Column | Meaning |
| --- | --- |
| `k`, `v` | Patent identifier/details |
| `s_products` | Canonical product SMILES |
| `s_reactants` | Reference reactant SMILES |
| `s_reagents` | Reference reagent SMILES |
| `s_solvents` | Reference solvent SMILES |
| `clean_response` | A Python-literal-compatible parsed patent record |

For product-only prediction, `predict_target_mols.py` also accepts a headerless CSV whose first column contains product SMILES.

## Configuration

Copy the environment template and fill in credentials for the OpenAI/LiteLLM-compatible endpoints used in your deployment:

```bash
cp .env.example .env
```

| Variables | Role |
| --- | --- |
| `SOTA_MODEL`, `SOTA_API_KEY`, `SOTA_BASE_URL`, `SOTA_TEMPERATURE` | Final pathway refinement and recipe generation |
| `RERANK_MODEL`, `RERANK_API_KEY`, `RERANK_BASE_URL`, `RERANK_TEMPERATURE` | Semantic reranking |
| `TEST_API_KEY`, `TEST_BASE_URL`, `TEST_JUDGE_MODEL_GPT` | GPT/Responses-API LLM judge |
| `TEST_JUDGE_MODEL`, `TEST_JUDGE_TEMPERATURE` | Chat-Completions LLM judge |
| `TEST_FAITHFULNESS_MODEL`, `TEST_FAITHFULNESS_TEMPERATURE` | Faithfulness judge |
| `LLM_SERVER_URL` | Local SFT MCP endpoint; default `http://localhost:8000/mcp` |
| `LLM_BASE_URL` | Optional OpenAI-compatible local-model endpoint; default `http://localhost:8000/v1` |

Do not commit `.env` or API credentials.

## Start the local pathway model

Start this service once in a dedicated terminal before running SynR-RAG:

```bash
cd /path/to/Agentic-SynR-RAG
conda activate synr-sft-server

python mcp_tools/local_llm_mcp.py \
  --model_path ./checkpoints/<local-sft-checkpoint>
```

The service listens on port `8000`. The main process starts the reaction-retrieval MCP tool as a subprocess, so no separate retrieval-server command is required for `run_app.py`.

## Batch Top-10 inference

`run_app.py` is the authoritative entry point for benchmark runs and ablations. The following production pattern splits the first 150 records into ten non-overlapping shards. Each shard uses its own output and checkpoint paths.

```bash
cd /path/to/Agentic-SynR-RAG
conda activate agentic-synr-rag

# Confirm that this checkout contains the intended embedding implementation.
grep -n "RXN embedding temp root" mcp_tools/database_embedding.py

RUN_DIR="./evaluate/$(date +%Y%m%d_%H%M%S)_gemini25_top10_10shards_policy"
mkdir -p "$RUN_DIR"
echo "$RUN_DIR" | tee "$RUN_DIR/RUN_DIR.txt"

for sid in $(seq 0 9); do
  mkdir -p "$RUN_DIR/shard_${sid}"
  nohup python run_app.py \
    -f "./evaluate/agent_benchmark_162_clean.csv" \
    -s "$RUN_DIR/shard_${sid}/01_gemini25_top10_model" \
    -p "$RUN_DIR/shard_${sid}/01_gemini25_top10_" \
    --num_samples 10 \
    --shard_id "$sid" \
    --num_shards 10 \
    --total_records 150 \
    --max_concurrency 1 \
    >> "$RUN_DIR/shard_${sid}/01_gemini25_top10.log" 2>&1 &
done

echo "started 10 shards -> $RUN_DIR"
jobs -l
```

Before starting a replacement run, inspect old jobs with `pgrep -af 'python.*run_app.py'` and terminate only the confirmed stale process IDs. Never let two jobs write to the same output prefix.

Each shard produces:

- `01_gemini25_top10_model.pkl`: a stream of pickled records
- `01_gemini25_top10_model.json`: concatenated indented JSON objects
- `01_gemini25_top10__checkpoint.json`: resumable progress, removed after successful completion
- `01_gemini25_top10.log`: redirected process log

The `.json` output is concatenated JSON, not one JSON array. Repository evaluation utilities parse this format directly.

## Results and qualitative examples

The README is part of the material examined by paper reviewers, but it should provide a compact, auditable result view rather than duplicate every raw run file. Put publication-facing artifacts under [`examples/`](./examples/) and use the following separation:

| Artifact | README role | Recommendation |
| --- | --- | --- |
| `gemini25_top10_display.json` | Main SynR-RAG qualitative output | **Required.** Link the complete curated file and show one compact example in the README. |
| `native_top1_display.json` | Direct native-LLM comparison | **Recommended.** Use the same target molecules/record IDs as the main display. |
| `generate_only_top1_display.json` | Retrieval-free ablation output | Optional as a raw display file. The quantitative generate-only ablation result should still appear in the benchmark summary. |

The main display should retain enough information to audit the pipeline:

- record ID and target product SMILES;
- ranked SFT Llama pathways;
- retrieved/reranked precedent summaries and citations;
- final SynR-RAG recipes;
- pathway identifiers and validation status.

Remove internal traces, duplicate payloads, API metadata, chain-of-thought, and secrets. If examples are manually selected, state the selection rule and record IDs; otherwise reviewers may reasonably interpret them as cherry-picked. Native and generate-only comparisons must use exactly the same frozen targets as the main display.

For the README itself, show:

1. one concise side-by-side qualitative case (SFT Llama, Native LLM, Generate-only, and SynR-RAG where available);
2. the final benchmark table for the full test set;
3. links to the complete curated JSON artifacts.

Do not paste full Native or Generate-only JSON into the README. Native is most useful as a matched qualitative control, while Generate-only is primarily an ablation and can be represented by its aggregate metrics plus an optional JSON link.

For a side-by-side case, compare rank 1 from every method. The remaining SynR-RAG Top-10 candidates may be shown separately to illustrate ranking and coverage, but they must not be presented as if Native or Generate-only had been evaluated at the same Top-10 setting unless those methods were actually run that way.

This repository ignores general run JSON files. Curated files placed directly under `examples/` are explicitly allowed by `.gitignore`.

### What “Top-10” means in the current implementation

The current local pathway generator does **not** make one native `num_beams=10` call. It uses this schedule:

| Requested k | Beam-search calls |
| --- | --- |
| 1 | `[1]` |
| 3 | `[1, 3]` |
| 5 | `[1, 3, 5]` |
| 10 | `[1, 3, 5, 10]` |

Candidates from all calls are canonicalized, chemically checked, pooled, deduplicated, reranked, and then truncated to the requested k. Consequently:

- Top-1/3/5 taken from a Top-10 run are prefixes of the **pooled Top-10 ranking**.
- They are not necessarily identical to separately generated native beam-1/3/5 results.
- To compare independent generation settings, run `--num_samples 1`, `3`, `5`, and `10` separately.

## Repeated target-molecule stability test

Prepare a CSV containing the frozen target set:

```csv
smiles
CC1=CC=C(C)C(NC(C2=NC(C=CC=C3)=C3C=C2)=O)=C1
CC1=CC=CC(C2=NC(C=CC=C3)=C3C(N2)=O)=C1
CNC([C@H](CC1=CNC2=C1C=CC=C2)NC(NC3=CC=CC(Cl)=C3)=O)=O
O=C(N(CC1)CCC1C2=CNC3=CC=CC=C32)C4=CN=C(C=CC=C5)C5=C4
```

Save it as `target_mols/stability_targets.csv`, then run three independent Top-10 predictions. The local SFT service described above must already be running.

```bash
cd /path/to/Agentic-SynR-RAG
mkdir -p ./output/stability

for r in 1 2 3; do
  python predict_target_mols.py \
    --input_csv ./target_mols/stability_targets.csv \
    --output_jsonl "./output/stability/gemini25_top10_rep${r}.jsonl" \
    --checkpoint_file "./output/stability/gemini25_top10_rep${r}.checkpoint.json" \
    --num_samples 10 \
    2>&1 | tee "./output/stability/gemini25_top10_rep${r}.log"
done
```

Use a new output and checkpoint filename for each replicate. The local SFT stage is deterministic beam search, while external reranking and refinement endpoints may still vary between runs.

## Extract Top-1 from a completed Top-10 run

The following script merges ten concatenated-JSON shards by `idx`, keeps rank 1, truncates aligned pathway fields, and writes both JSON and pickle outputs. This produces rank 1 from the pooled Top-10 run; it does not recreate an independent `--num_samples 1` run.

```bash
SRC="evaluate/20260811_110143_gemini25_top10_10shards_policy"
OUT="evaluate/policy_top1_from_top10_0811_110143"
mkdir -p "$OUT"
export SRC OUT

python - <<'PY'
import glob
import json
import os
import pickle

src = os.environ["SRC"]
out = os.environ["OUT"]
out_json = os.path.join(out, "gemini25_top1_from_top10.json")
out_pkl = os.path.join(out, "gemini25_top1_from_top10.pkl")

def parse_concat(path):
    text = open(path, encoding="utf-8").read()
    decoder, pos, records = json.JSONDecoder(), 0, []
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        obj, pos = decoder.raw_decode(text, pos)
        records.append(obj)
    return records

by_idx = {}
pattern = os.path.join(src, "shard_*", "01_gemini25_top10_model.json")
for path in sorted(glob.glob(pattern)):
    for record in parse_concat(path):
        idx = record.get("idx")
        if not isinstance(idx, int):
            continue
        for key in ("full_recipe", "pathways"):
            value = record.get(key)
            if isinstance(value, list):
                record[key] = value[:1]
        for key in (
            "final_reports", "pathway_id", "top5_retrieval", "retrieval_first",
            "top3_rerank_extract", "top3_rerank_retrieval", "history",
        ):
            value = record.get(key)
            if isinstance(value, list) and len(value) > 1:
                record[key] = value[:1]
        by_idx[idx] = record

records = [by_idx[idx] for idx in sorted(by_idx)]
if records:
    print("records:", len(records), "idx:", records[0]["idx"], "-", records[-1]["idx"])
else:
    raise RuntimeError(f"No records found under {src}")

with open(out_json, "w", encoding="utf-8") as handle:
    for record in records:
        json.dump(record, handle, ensure_ascii=False, indent=4)
        handle.write("\n")

with open(out_pkl, "wb") as handle:
    for record in records:
        pickle.dump(record, handle)

print("wrote", out_json, out_pkl)
PY
```

For a smaller standard JSON array that retains all recipes, use:

```bash
python benchmark/slim_run_outputs.py \
  --input_dir "$SRC" \
  --output "$SRC/slim_all.json"
```

## Evaluation

### Rule-based Top-k metrics

```bash
python benchmark/evaluate_reactant_metrics.py \
  --input_dir ./evaluate/20260811_110143_gemini25_top10_10shards_policy \
  --output_prefix ./evaluate/20260811_110143_gemini25_top10_10shards_policy/reactant_metrics
```

The report calculates Top-1, Top-3, Top-5, and Top-10 results for:

- **Reactant accuracy**
- **Molecular validity**
- **Elemental consistency**
- **Structural compatibility**

The script’s output labels map to the paper terminology as follows: `local` = **SFT Llama**, `agent` = **Syn-RRAG**, and `agent+alt` = **Syn-RRAG with eligible alternative reactants**. Top-k is computed over prefixes of the final ranked output, not by rerunning the generator at each k.

### LLM-as-a-judge for extracted Top-1

```bash
mkdir -p ./benchmark/policy_top1_from_top10_0811_110143

python benchmark/prepare_llm_judge_dataset.py \
  --input_file ./evaluate/policy_top1_from_top10_0811_110143/gemini25_top1_from_top10.json \
  --prefix ./benchmark/policy_top1_from_top10_0811_110143/gemini25_agentic_top1_

python benchmark/test_API_with_self_correction_gpt.py \
  --input_file ./benchmark/policy_top1_from_top10_0811_110143/gemini25_agentic_top1_llm_judge_eval_results.json \
  --prefix ./benchmark/policy_top1_from_top10_0811_110143/gemini25_agentic_top1_
```

### Faithfulness

```bash
python benchmark/test_API_with_faithfulness_score.py \
  --input_file ./benchmark/policy_top1_from_top10_0811_110143/gemini25_agentic_top1_llm_judge_eval_results.json \
  --prefix ./benchmark/policy_top1_from_top10_0811_110143/gemini25_faithfulness_top1_
```

Use `--test_offline` with the faithfulness script only when the corresponding audit CSV already exists and you want to recompute its summary without new API calls.

## Baselines and ablations

### Native LLM baseline

This baseline asks the configured main model to generate a complete recipe directly, without the local SFT pathway generator or retrieval pipeline.

```bash
mkdir -p ./native ./benchmark/native_gemini25_top1_0809

python OSCAR_generate_only_native.py \
  -f ./evaluate/agent_benchmark_162_clean.csv \
  -s ./native/gemini25_mode_top1_0809 \
  -p ./native/gemini25_top1_0809

python benchmark/prepare_llm_judge_dataset.py \
  --input_file ./native/gemini25_mode_top1_0809.csv \
  --prefix ./benchmark/native_gemini25_top1_0809/gemini25_native_top1_

python benchmark/test_API_with_self_correction_gpt.py \
  --input_file ./benchmark/native_gemini25_top1_0809/gemini25_native_top1_llm_judge_eval_results.json \
  --prefix ./benchmark/native_gemini25_top1_0809/gemini25_native_top1_
```

`OSCAR_generate_only_native.py` historically writes concatenated JSON records using a `.csv` filename; the preparation utility handles the content as records despite that extension.
The current native runner reads `SOTA_MODEL` and `SOTA_API_KEY` but uses `https://www.litellm.org/` as its fixed gateway URL.

### Generate-only ablation

This mode retains the local SFT pathway proposal and final generation but removes retrieval and semantic reranking.

```bash
python run_app.py \
  -f ./evaluate/agent_benchmark_162_clean.csv \
  -s ./evaluate/generate_only/gemini25_top1_0810 \
  -p ./evaluate/generate_only/gemini25_top1_0810 \
  --generate_only \
  --num_samples 1

mkdir -p ./benchmark/generate_only

python benchmark/prepare_llm_judge_dataset.py \
  --input_file ./evaluate/generate_only/gemini25_top1_0810.json \
  --prefix ./benchmark/generate_only/gemini25_top1_0810_

python benchmark/test_API_with_self_correction_gpt.py \
  --input_file ./benchmark/generate_only/gemini25_top1_0810_llm_judge_eval_results.json \
  --prefix ./benchmark/generate_only/judge
```

For the structural-retrieval-only ablation, replace `--generate_only` with `--simple_rag`.

## Product-only prediction

Use `predict_target_mols.py` when ground-truth patent fields are unavailable:

```bash
python predict_target_mols.py \
  --input_csv ./target_mols/target_mol.csv \
  --output_jsonl ./target_mols/target_mol_top10_predictions.jsonl \
  --num_samples 10
```

Use `--smiles_column <column>` for a headered CSV, or omit it for a headerless file whose first column contains product SMILES.

## Reproducibility notes

- The local SFT service uses deterministic beam search (`do_sample=False`). Therefore caller-supplied `temperature`, `top_p`, and `top_k` values are ignored; differences between Top-1/3/5/10 come from beam width and candidate pooling, not sampling randomness.
- No global random seed is set by the repository. GPU kernels and external model endpoints may still introduce variation.
- The GPT Responses-API judge intentionally does not send a temperature because reasoning models such as GPT-5 Pro may reject or ignore it.
- Record the code revision, `.env` model identifiers, endpoint/provider versions, asset versions, run directory, and logs for every reported experiment.
- For judge stability, evaluate the same frozen prepared dataset multiple times and report the aggregation procedure; do not compare scores from differently prepared inputs.

## Execution entry points and current limitations

- Use `run_app.py` for benchmark CSVs, sharding, and paper ablations.
- Use `predict_target_mols.py` for product-only inference; it replaces the legacy single-molecule port runner and supports full RAG, simple RAG, and generate-only modes.
- Both entry points start `mcp_tools/mcp_rag.py` internally, but the local SFT service on port 8000 must already be running.
- Failed validation pathways may be excluded from `full_recipe`. Inspect `validation_errors` and pathway identifiers rather than assuming all output lists remain positionally aligned after filtering.

## Citation

Citation information will be added with the accompanying manuscript. If this repository contributes to published work, please cite the released paper and this software repository.

## License

This project is released under the [Apache License 2.0](./LICENSE).
