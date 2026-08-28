# Agentic-Syn-RRAG

Agentic-Syn-RRAG is the release repository for **Syn-RRAG**, a retrieval-augmented framework for single-step retrosynthesis and synthesis-protocol generation. Given a target product SMILES, the system combines a locally served SFT model, RXNGraphormer/FAISS precedent retrieval, LLM reranking, and LLM-guided protocol refinement.

This repository contains the inference and evaluation runtime, a frozen 150-target benchmark, curated outputs, and a patch for reproducing the SFT training workflow with LLaMA-Factory.

> **Research-use warning**
> Generated routes and procedures are model outputs, not experimentally validated instructions. They must be reviewed by qualified chemists before use.

## Contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Models and data](#models-and-data)
- [SFT training](#sft-training)
- [Configuration](#configuration)
- [Quick start](#quick-start)
- [Batch inference](#batch-inference)
- [Benchmark and evaluation](#benchmark-and-evaluation)
- [Baselines and ablations](#baselines-and-ablations)
- [Reproducibility and limitations](#reproducibility-and-limitations)
- [Citation](#citation)

## Overview

Syn-RRAG implements a **Generate–Retrieve–Refine** workflow:

1. **Generate** — the local SFT model proposes candidate reactants and initial reagents/solvents.
2. **Retrieve** — RXNGraphormer embeds each proposed reaction and FAISS returns ten structurally similar patent precedents.
3. **Rerank** — the configured reranking model scores the retrieved precedents and retains the three most relevant records.
4. **Refine** — the configured main model reviews the pathway and evidence, applies output validation, and writes a structured protocol.

The ten retrieved precedents in step 2 are distinct from the final Top-10 pathway candidates produced when `--num_samples 10` is used.

<p align="center">
  <img src="./images/Syn-RRAG.jpg" alt="Syn-RRAG architecture" width="760">
</p>

### Key features

The runtime supports:

- full Syn-RRAG inference;
- Top-k pathway generation with canonicalization, chemical checks, deduplication, and reranking;
- product-only prediction and benchmark CSV input;
- deterministic index-based sharding with resumable checkpoints;
- retrieval-free, simple-RAG, and native-LLM comparisons;
- rule-based reactant metrics, protocol judging, and faithfulness evaluation.

## Repository structure

```text
.
├── OSCAR_main.py                      # LangGraph workflow construction
├── run_app.py                         # Benchmark, sharding, and ablation runner
├── predict_target_mols.py             # Product-only Top-k inference
├── OSCAR_generate_only_native.py      # Direct native-LLM baseline
├── nodes/
│   ├── rag_node.py                    # Retrieval, reranking, refinement, validation
│   ├── rag_node_simple.py             # Retrieval without semantic reranking
│   └── generate_only_node.py          # Retrieval-free refinement
├── mcp_tools/
│   ├── local_llm_mcp.py               # Local SFT model service
│   ├── self_refine_loop_agent.py      # Candidate generation and verification
│   ├── mcp_rag.py                     # Runtime retrieval/refinement MCP tool
│   ├── database_embedding.py          # RXNGraphormer query embedding and search
│   ├── save_load_embedding.py         # FAISS/database loading and persistence
│   └── retrieval_sys.py               # Offline index-building utility
├── utils/reaction_plausibility.py     # Element and scaffold checks
├── benchmark/                         # Extraction and evaluation scripts
├── evaluate/test_150.csv              # Frozen 150-target benchmark
├── examples/                          # Curated, publication-facing outputs
├── target_mols/target_mol.csv         # Three product-only example targets
├── train_examples/                    # LLaMA-Factory patch and requirements
├── pretrained_classification_model/   # RXNGraphormer model location
├── requirements_main.txt              # Main runtime environment
├── requirements_server.txt            # Local SFT service environment
└── .env.example                       # Endpoint/model configuration template
```

Large model weights, the reaction database, and its FAISS index are not included.

## Installation

The main pipeline and the local SFT service use separate environments because their PyTorch and Transformers requirements differ. The commands below assume Bash; adapt activation and line-continuation syntax on other shells.

### 1. Clone the repository

```bash
git clone https://github.com/SamSamChu/Agentic-Syn-RRAG.git
cd Agentic-Syn-RRAG
```

### 2. Install the main runtime

Python 3.10 is recommended.

```bash
conda create -n agentic-syn-rrag python=3.10 -y
conda activate agentic-syn-rrag

pip install -r requirements_main.txt \
  -f https://data.pyg.org/whl/torch-2.2.0+cpu.html \
  --extra-index-url https://download.pytorch.org/whl/cpu

git clone -b pytorch2 https://github.com/licheng-xu-echo/RXNGraphormer.git
pip install ./RXNGraphormer
```

### 3. Install the local SFT service

Python 3.12 is recommended for the versions pinned in `requirements_server.txt`.

```bash
conda create -n syn-rrag-sft-server python=3.12 -y
conda activate syn-rrag-sft-server
pip install -r requirements_server.txt
```

The local checkpoint is loaded in bfloat16 with `device_map="auto"`; a compatible accelerator and sufficient memory are therefore expected for normal use.

## Models and data

Arrange the external runtime assets as follows:

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
    └── test_150.csv                    # included
```

The RXNGraphormer classification checkpoint is available from [Figshare](https://doi.org/10.6084/m9.figshare.28356077). Extract it into `pretrained_classification_model/`.

The reaction database, FAISS index, and `indices_map_update.npy` must be generated from the same ordered dataset. Mixing an index or index map from a different database ordering produces invalid retrieval results. This snapshot does not specify a public download URL for that matched asset set; obtain it from the project release or build it from an authorized reaction corpus. `mcp_tools/retrieval_sys.py` contains the offline index-building path; review its input and output paths before running it.

Place a Hugging Face-compatible SFT checkpoint under `checkpoints/`. It may be produced with the training procedure below or supplied separately by the project authors.

## SFT training

The SFT checkpoint is a prerequisite for the **Generate** stage. If you do not have a checkpoint distributed by the project authors, train and export one before proceeding to [Configuration](#configuration) and [Quick start](#quick-start).

The training release is a patch against a fixed LLaMA-Factory revision, not a standalone trainer. All commands in this section start in `train_examples/`.

### 1. Prepare LLaMA-Factory

```bash
cd /path/to/Agentic-SynR-RAG/train_examples
git clone https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory

git fetch origin 2b27283ba0566eda9ec7ac335642807189c87e70
git checkout 2b27283ba0566eda9ec7ac335642807189c87e70
git apply --check ../my_changes.patch
git apply ../my_changes.patch
```

The clone directory is `LlamaFactory`. The pinned commit is required because the patch modifies internal data loading, SFT workflow, and trainer files in addition to adding configs and evaluation utilities.

### 2. Install training dependencies

The supplied training requirements target Python 3.12 and CUDA 12.8:

```bash
conda create -n syn-rrag-train python=3.12 -y
conda activate syn-rrag-train

pip install .
pip install -r ../requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

Verify the actual environment before launching a long run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 3. Prepare model and datasets

Set `model_name_or_path` in the patched YAML files to a local Llama-3-8B-Instruct-compatible checkpoint. The paths included in the patch are environment-specific examples and must be changed.

The provided YAML files expect:

```text
LlamaFactory/
├── wipo_data/
│   ├── dataset_info.json
│   ├── train.jsonl
│   └── valid.jsonl
└── data_uspto50/
    ├── dataset_info.json
    ├── train_50k_class.jsonl
    └── valid_50k_class.jsonl
```

The patch adds dataset metadata templates under `data_wipo/` and `data_uspto50k/`. Copy or link the relevant template into the runtime dataset directory, or update `dataset_dir` in the YAML. Training and evaluation corpora are not bundled.

### 4. Run training and SFT evaluation

From the `LlamaFactory` root:

```bash
# WIPO multi-task training
bash train_wipo.sh

# USPTO-50K retrosynthesis/forward-prediction training
bash train_uspto.sh

# Patched WIPO and USPTO-50K evaluation instructions
cd reaction_eval
```

Review `wipo_multi_task_full.yaml` and `uspto_50k_class.yaml` before running, especially model paths, dataset paths, output directories, precision, batch size, and distributed settings. The reference WIPO run is intended for a multi-GPU setup; memory and throughput depend on local hardware.

After export, place the Hugging Face checkpoint under `checkpoints/<local-sft-checkpoint>/`. The inference sections below use this checkpoint through `mcp_tools/local_llm_mcp.py`.

## Configuration

Copy the template and replace placeholder credentials:

```bash
cp .env.example .env
```

| Variable | Used for | Default in code |
| --- | --- | --- |
| `SOTA_MODEL` | Final pathway review and protocol generation | `gemini-2.5-pro` |
| `SOTA_API_KEY` | Main-model API credential | none |
| `SOTA_BASE_URL` | OpenAI-compatible main-model endpoint | `https://www.litellm.org/` |
| `SOTA_TEMPERATURE` | Main-model temperature | `1.0` |
| `RERANK_MODEL` | Patent-precedent reranking | `gemini-2.5-flash` |
| `RERANK_API_KEY` | Reranking-model credential | none |
| `RERANK_BASE_URL` | OpenAI-compatible reranking endpoint | `https://www.litellm.org/` |
| `RERANK_TEMPERATURE` | Reranking temperature | `1.0` |
| `LLM_SERVER_URL` | Local SFT MCP endpoint | `http://localhost:8000/mcp` |
| `TEST_API_KEY`, `TEST_BASE_URL` | Evaluation-model credential and endpoint | endpoint defaults to LiteLLM |
| `TEST_JUDGE_MODEL_GPT` | Protocol judge | `gpt-5-pro` |
| `TEST_FAITHFULNESS_MODEL` | Faithfulness judge | `gpt-5-pro` |
| `TEST_FAITHFULNESS_TEMPERATURE` | Faithfulness-judge temperature | `1.0` in code |

`LLM_BASE_URL` is retained in `self_refine_loop_agent.py` for an OpenAI-compatible local client, but the current generation path communicates through `LLM_SERVER_URL`. The native baseline is also a legacy path: it reads `SOTA_MODEL` and `SOTA_API_KEY` but currently uses a fixed LiteLLM gateway URL.

Do not commit `.env` or API credentials.

## Quick start

### 1. Start the local pathway service

Run the service in a dedicated terminal:

```bash
cd /path/to/Agentic-SynR-RAG
conda activate syn-rrag-sft-server

python mcp_tools/local_llm_mcp.py \
  --model_path ./checkpoints/<local-sft-checkpoint>
```

The service listens on port `8000`. The inference entry points start `mcp_tools/mcp_rag.py` as a subprocess, so the retrieval tool does not require a second manually started service.

### 2. Predict synthesis plans from product SMILES

The bundled input is a headerless CSV containing three targets:

```bash
conda activate agentic-syn-rrag

python predict_target_mols.py \
  -f target_mols/target_mol.csv \
  --num_samples 10 \
  -o target_mols/target_mol_top10_predictions.jsonl
```

For a headered CSV, pass `--smiles_column <column>`. Without that option, the script recognizes common names including `smiles`, `target_smiles`, `product_smiles`, and `s_products`. Every input SMILES is validated and canonicalized with RDKit before inference.

The output contains one pretty-printed JSON object per target. A checkpoint named `<output>.checkpoint.json` is maintained during execution and removed after all targets complete successfully. If a run is interrupted, rerunning the same command resumes from the checkpoint or existing output.

Optional modes:

```bash
# SFT proposal + final generation, without retrieval or reranking
python predict_target_mols.py -f target_mols/target_mol.csv --generate_only

# Structural retrieval without semantic reranking
python predict_target_mols.py -f target_mols/target_mol.csv --simple_rag
```

`--condition_sampling` affects only local SFT reagent/solvent generation. Reactant generation remains deterministic beam search. Its sampling controls are `--condition_temperature`, `--condition_top_p`, and `--condition_top_k`.

## Batch inference

`run_app.py` is the benchmark and ablation entry point. Its input CSV must contain `k`, `v`, `s_products`, `s_reactants`, `s_reagents`, `s_solvents`, and a Python-literal-compatible `clean_response` field. The included `evaluate/test_150.csv` has 150 rows and the required columns.

A single-process run is:

```bash
python run_app.py \
  -f ./evaluate/test_150.csv \
  -s ./evaluate/syn_rrag_top10/model \
  -p ./evaluate/syn_rrag_top10/run \
  --num_samples 10 \
  --max_concurrency 1
```

For ten deterministic, non-overlapping shards:

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)_syn_rrag_top10"
RUN_DIR="./evaluate/${RUN_ID}"
OUTPUT_STEM="01_gemini25_top10"
mkdir -p "$RUN_DIR"
printf '%s\n' "$OUTPUT_STEM" > "$RUN_DIR/OUTPUT_STEM.txt"

for sid in $(seq 0 9); do
  SHARD_DIR="$RUN_DIR/shard_${sid}"
  mkdir -p "$SHARD_DIR"
  nohup python run_app.py \
    -f ./evaluate/test_150.csv \
    -s "$SHARD_DIR/${OUTPUT_STEM}_model" \
    -p "$SHARD_DIR/${OUTPUT_STEM}_" \
    --num_samples 10 \
    --shard_id "$sid" \
    --num_shards 10 \
    --total_records 150 \
    --max_concurrency 1 \
    > "$SHARD_DIR/${OUTPUT_STEM}.log" 2>&1 &
done
```

Do not allow two processes to write to the same output prefix. Each shard writes:

- `${OUTPUT_STEM}_model.pkl`: a stream of pickled records;
- `${OUTPUT_STEM}_model.json`: pretty-printed, concatenated JSON objects, not a JSON array;
- `${OUTPUT_STEM}__checkpoint.json`: resumable progress, removed after that shard completes;
- `${OUTPUT_STEM}.log`: redirected process output from the command above.

## Benchmark and evaluation

### Reported benchmark

The accompanying manuscript evaluates the full pipeline on 150 targets sampled from the product-disjoint WIPO-2M held-out test set. The release snapshot includes this frozen subset at [`evaluate/test_150.csv`](./evaluate/test_150.csv).

| Track | Evaluation setting | Reported result |
| --- | --- | --- |
| Standalone SFT backbone | WIPO-2M held-out test (~10k products) | Top-1/3/5/10: 44.5% / 65.1% / 70.8% / 75.9% |
| Full Syn-RRAG | 150 targets, Top-1 | Reactant accuracy: **48.0%** |
| SFT Llama | Same 150 targets | Reactant accuracy: 46.0% |
| Native Gemini-2.5-Pro | Same 150 targets | Reactant accuracy: 14.7% |
| Full Syn-RRAG rule checks | Same 150 targets, Top-1 | Validity: 100.0%; elemental consistency: 100.0%; structural compatibility: 99.3% |

These values are reported results, not recomputed during installation. Use the commands below to evaluate a completed run.

### Metric definitions

| Metric | Implementation in this repository |
| --- | --- |
| Reactant accuracy | After RDKit canonicalization, every reference reactant component must appear in the pooled predicted reactants and reagents. The fields are pooled because a model may classify a true reactant as a reagent. |
| Molecular validity | Every predicted molecular component must parse and canonicalize with RDKit. |
| Elemental consistency | Every element type in the target product must occur in the predicted reactant/reagent pool; this is presence-only, not atom-count balancing. |
| Structural compatibility | Each substantial reactant fragment must satisfy the connected-MCS scaffold policy in `utils/reaction_plausibility.py`. |
| Full-protocol plausibility | An independent judge scores temperature/time, stoichiometry, solvent choice, and operational practicality from 0 to 10. |

In evaluation output, `local` denotes the SFT proposal, `agent` denotes Syn-RRAG, and `agent+alt` includes eligible alternative reactants. The latter is used only where the recorded reactant-revision policy allows it.

### Interpreting Top-k

The local generator pools results from milestone beam widths:

| Requested k | Beam-search calls |
| --- | --- |
| 1 | `[1]` |
| 3 | `[1, 3]` |
| 5 | `[1, 3, 5]` |
| 10 | `[1, 3, 5, 10]` |

Candidates are canonicalized, checked, pooled, deduplicated, sorted, and truncated to the requested k. Therefore, Top-1/3/5 metrics from a completed Top-10 run use prefixes of the final pooled ranking. They need not equal results from independent `--num_samples 1`, `3`, or `5` runs.

### Extract Top-1 and compute rule-based metrics

Set `RUN_DIR` to a completed sharded run and reuse its output stem:

```bash
OUTPUT_STEM="${OUTPUT_STEM:-01_gemini25_top10}"
SHARD_PATTERN="shard_*/${OUTPUT_STEM}_model.json"

python benchmark/extract_top1_from_top10.py \
  --input_dir "$RUN_DIR" \
  --pattern "$SHARD_PATTERN"

mkdir -p "$RUN_DIR/metrics"
python benchmark/evaluate_reactant_metrics.py \
  --input_dir "$RUN_DIR" \
  --pattern "$SHARD_PATTERN" \
  --output_prefix "$RUN_DIR/metrics/reactant"
```

Top-1 files are written to `RUN_DIR/derived/top1_from_top10/`. Metric outputs are `<prefix>_summary.json` and `<prefix>_per_record.csv`.

### Protocol judge and faithfulness

```bash
TOP1_FILE="$RUN_DIR/derived/top1_from_top10/top1_from_top10.json"
JUDGE_DIR="./benchmark/${RUN_ID}/top1_from_top10"
JUDGE_PREFIX="$JUDGE_DIR/syn_rrag_top1_"
mkdir -p "$JUDGE_DIR"

python benchmark/prepare_llm_judge_dataset.py \
  --input_file "$TOP1_FILE" \
  --prefix "$JUDGE_PREFIX"

python benchmark/test_API_with_self_correction_gpt.py \
  --input_file "${JUDGE_PREFIX}llm_judge_eval_results.json" \
  --prefix "$JUDGE_PREFIX"

python benchmark/test_API_with_faithfulness_score.py \
  --input_file "${JUDGE_PREFIX}llm_judge_eval_results.json" \
  --prefix "$JUDGE_DIR/syn_rrag_faithfulness_top1_"
```

Use `--test_offline` with the faithfulness script only when its audit CSV already exists and only the summary should be recomputed.

### Curated outputs

The release includes redacted, publication-facing examples:

| File | Contents |
| --- | --- |
| [`examples/gemini25_top10_display.json`](./examples/gemini25_top10_display.json) | Full Syn-RRAG pathways, retrieved/reranked evidence, and final recipes |
| [`examples/native_top1_display.csv`](./examples/native_top1_display.csv) | Native-LLM Top-1 records; despite the extension, this is concatenated JSON content |
| [`examples/generate_only_top1_display.json`](./examples/generate_only_top1_display.json) | Retrieval-free Top-1 outputs |

The files contain the same 150 targets in the same `idx` order, but their fields differ by experimental setting. API metadata, hidden reasoning, and credentials are excluded.

## Baselines and ablations

### Native LLM

This path asks the configured main model for a complete protocol without the local SFT generator or retrieval:

```bash
python OSCAR_generate_only_native.py \
  -f ./evaluate/test_150.csv \
  -s ./native/gemini25_top1/model \
  -p ./native/gemini25_top1/run
```

The runner writes concatenated JSON records to a `.csv`-named file and a pickle stream. `prepare_llm_judge_dataset.py` handles this legacy format.

### Syn-w/o-RRAG

This ablation retains SFT pathway proposal and final protocol generation but removes precedent retrieval and semantic reranking:

```bash
python run_app.py \
  -f ./evaluate/test_150.csv \
  -s ./evaluate/generate_only_top1/model \
  -p ./evaluate/generate_only_top1/run \
  --generate_only \
  --num_samples 1
```

### Simple RAG

`--simple_rag` uses the structural retrieval results directly and omits semantic reranking. It is a code-supported diagnostic mode rather than the full Syn-RRAG setting.

## Reproducibility and limitations

- Reactant generation defaults to deterministic beam search (`do_sample=False`); condition sampling is opt-in and does not change that behavior.
- No repository-wide random seed is set. GPU kernels and external model providers may still introduce variation.
- The main benchmark runner processes records sequentially inside each shard; `--max_concurrency` configures LangGraph work within a record.
- External model IDs and provider implementations can change. Record the code revision, endpoint/provider version, model IDs, asset versions, run directory, and logs.
- Failed or warned validation paths are recorded in `validation_errors`. Inspect pathway IDs instead of assuming every result list is positionally aligned after filtering.
- The rule-based checks are screening heuristics, not reaction simulation or experimental validation.
- Judge scores depend on the prepared dataset and judge endpoint. Compare only runs evaluated from the same frozen inputs and report repeated-run aggregation where applicable.

## Citation

Citation metadata will be added with the accompanying manuscript. Until then, cite both the manuscript and this repository when using the released code, benchmark, or examples.

## License

This snapshot does not currently include a standalone `LICENSE` file. Do not assume redistribution or reuse terms; contact the repository maintainers for licensing information.
