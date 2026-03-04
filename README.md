# pi-crop-ai

An offline-first, edge-deployed AI agent for crop disease diagnosis and treatment planning, designed to run on a **Raspberry Pi 5**. The agent combines a CNN-based vision model (TFLite), a locally-served LLM (Ollama / HuggingFace), and a FAISS vector memory store to diagnose diseases from camera images, retrieve similar historical cases, and generate actionable treatment plans — all without an internet connection.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Agent Pipeline — Inputs & Outputs](#agent-pipeline--inputs--outputs)
3. [Folder Structure](#folder-structure)
4. [Where to Place Model Files](#where-to-place-model-files)
5. [Configuration Files](#configuration-files)
6. [Key Components](#key-components)
7. [Memory System](#memory-system)
8. [Prompts](#prompts)
9. [Setup & Installation](#setup--installation)
10. [Running the Agent](#running-the-agent)
11. [Seeding Knowledge Memory](#seeding-knowledge-memory)
12. [Testing](#testing)
13. [Training the CNN](#training-the-cnn)
14. [Deployment on Raspberry Pi](#deployment-on-raspberry-pi)
15. [What Is Not Yet Implemented](#what-is-not-yet-implemented)

---

## Architecture Overview

```
Camera / Sensors
       │
       ▼
┌─────────────────────┐
│  Perception Layer   │  disease_detector.py (CNN / TFLite)
│                     │  severity_estimator.py
│                     │  camera.py
└────────┬────────────┘
         │  case_context dict
         ▼
┌─────────────────────┐
│  DecisionAgent      │  crop_agent.py  ← main orchestrator
│  (crop_agent/)      │
│  ┌───────────────┐  │
│  │ VectorStore   │  │  FAISS semantic search over historical cases
│  │ MetadataStore │  │  SQLite case text store
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │  LLMEngine    │  │  Ollama (qwen2.5:1.5b) or HuggingFace backend
│  │  PromptRunner │  │
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │SafetyValidator│  │  Blocks unknown/unsafe actions
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │TreatmentPlanner  │  Generates step-by-step plan via LLM
│  └───────────────┘  │
└────────┬────────────┘
         │  result dict
         ▼
   { decision, reason, plan }
```

---

## Agent Pipeline — Inputs & Outputs

### Input: `case_context` dict

This is the primary input to `DecisionAgent.decide()`. All fields are required except `temperature` and `humidity`.

| Field | Type | Required | Description | Example |
|---|---|---|---|---|
| `crop` | `str` | Yes | Name of the crop | `"Tomato"` |
| `disease` | `str` | Yes | Diagnosed disease name | `"Leaf Blight"` |
| `severity` | `str` | Yes | One of `"low"`, `"medium"`, `"high"` | `"medium"` |
| `confidence` | `float` | Yes | CNN model confidence score (0.0–1.0) | `0.88` |
| `temperature` | `int` / `float` | No | Ambient temperature in °C from DHT22 | `32` |
| `humidity` | `int` / `float` | No | Relative humidity % from DHT22 | `78` |

**Example input:**
```python
case_context = {
    "crop": "Tomato",
    "disease": "Leaf Blight",
    "severity": "medium",
    "confidence": 0.88,
    "temperature": 32,
    "humidity": 78,
}
```

### Output: `result` dict

`DecisionAgent.decide()` returns a single dict containing three top-level keys.

| Key | Type | Description |
|---|---|---|
| `decision` | `str` | One of the allowed actions from `agent.yaml` (see below) |
| `reason` | `str` | Short English explanation produced by the LLM |
| `plan` | `list[dict]` | Ordered list of treatment steps (from `TreatmentPlanner`) |
| `safe` | `bool` | `True` if the decision passed safety validation |
| `override` | `bool` | `True` if the safety validator replaced the LLM decision |
| `requires_confirmation` | `bool` | Present when `require_human_confirmation: true` in config |

**Allowed `decision` values** (defined in `config/agent.yaml`):
- `NO_TREATMENT`
- `MONITOR_24H`
- `MONITOR_48H`
- `FERTILIZER`
- `MEDICINAL_TREATMENT`

Each `plan` step has this shape:

| Field | Type | Description |
|---|---|---|
| `step` | `int` | Sequential step number (1-based) |
| `action` | `str` | One of the allowed plan actions (see `agent.yaml`) |
| `details` | `str` | One-sentence English description of what to do |

**Allowed `plan` actions** (defined in `config/agent.yaml`):
`APPLY_FUNGICIDE`, `APPLY_FERTILIZER`, `INCREASE_IRRIGATION`, `REDUCE_IRRIGATION`, `PRUNE_AFFECTED_AREA`, `MONITOR_DAILY`, `MONITOR_WEEKLY`, `NOTIFY_AGRONOMIST`, `NO_ACTION`

**Example output:**
```json
{
  "decision": "MEDICINAL_TREATMENT",
  "reason": "Medium severity Leaf Blight with high humidity warrants fungicide application.",
  "safe": true,
  "override": false,
  "requires_confirmation": true,
  "plan": [
    {"step": 1, "action": "APPLY_FUNGICIDE", "details": "Apply copper-based fungicide to affected foliage."},
    {"step": 2, "action": "PRUNE_AFFECTED_AREA", "details": "Remove and dispose of visibly blighted leaves."},
    {"step": 3, "action": "MONITOR_DAILY", "details": "Check plant condition daily for the next 48 hours."},
    {"step": 4, "action": "NOTIFY_AGRONOMIST", "details": "Alert field agronomist if symptoms worsen."}
  ]
}
```

---

## Folder Structure

```
pi-crop-ai/
│
├── main.py                        # Entry point (currently minimal)
├── pyproject.toml                 # Project metadata and dependencies (uv / hatch)
├── README.md
│
├── config/                        # All YAML configuration files
│   ├── agent.yaml                 # Allowed agent actions, safety rules, planner actions
│   ├── database.yaml              # (reserved) Database config
│   ├── deployment.yaml            # Edge deployment settings (Pi, quantization)
│   ├── model.yaml                 # CNN model path, LLM backend, model name, token limits
│   ├── sensors.yaml               # DHT22 and camera sensor configuration
│   └── system_config.yaml         # Memory index path, embedding model, FAISS settings
│
├── crop_agent/                    # Core Python package
│   ├── utils.py                   # Shared utilities: load_yaml, JSON parsing, path constants
│   │
│   ├── agent/                     # Orchestration layer
│   │   ├── crop_agent.py          # DecisionAgent — main entry point for the pipeline
│   │   ├── planner.py             # TreatmentPlanner — builds step-by-step plans via LLM
│   │   ├── safety_validator.py    # SafetyValidator — blocks disallowed/unknown actions
│   │   └── tools.py               # (placeholder) Tool definitions for future agentic tools
│   │
│   ├── llm/                       # LLM abstraction layer
│   │   ├── llm_client.py          # LLMEngine — orchestrates backend + prompt execution
│   │   ├── model_backends.py      # OllamaBackend, HuggingFaceBackend, get_backend()
│   │   ├── prompt_loader.py       # build_prompt(), build_plan_prompt() — prompt builders
│   │   ├── prompt_runner.py       # PromptRunner — executes prompts against any backend
│   │   └── response_parser.py     # Thin wrapper around Utils.parse_response
│   │
│   ├── memory/                    # Persistent case memory
│   │   ├── embedder.py            # Embedder — SentenceTransformer wrapper (384-dim)
│   │   ├── vector_store.py        # VectorStore — FAISS index for semantic search
│   │   └── metadata_store.py      # MetadataStore — SQLite store for case text
│   │
│   ├── perception/                # Vision + severity (stubs — to be implemented)
│   │   ├── camera.py              # Camera capture from Pi Camera / USB
│   │   ├── disease_detector.py    # CNN/TFLite inference → disease label + confidence
│   │   └── severity_estimator.py  # Severity estimation from detection output
│   │
│   ├── sensors/                   # Hardware sensor drivers (stubs — to be implemented)
│   │   ├── dht_sensor.py          # DHT22 temperature & humidity reader
│   │   ├── sensor_fusion.py       # Merges sensor readings into a single context dict
│   │   └── soil_sensor.py         # (reserved) Soil moisture sensor
│   │
│   └── evaluation/                # Benchmarking (stubs — to be implemented)
│       ├── agent_evaluator.py     # End-to-end accuracy evaluation
│       └── latency_profiler.py    # Latency benchmarking for on-device inference
│
├── data/
│   ├── seed/
│   │   └── sample.json            # Seed cases for initial memory ingestion
│   ├── memory/
│   │   └── crop_memory.index      # FAISS index file (auto-generated, do not commit)
│   │   └── cases.db               # SQLite metadata DB (auto-generated, do not commit)
│   ├── raw/                       # Raw training images (organised by class)
│   ├── processed/                 # Preprocessed/augmented images for training
│   ├── annotations/               # Label files for training data
│   └── benchmarks/                # Benchmark result logs
│
├── models/
│   ├── cnn/                       # ← Place TFLite model here (see below)
│   ├── embeddings/                # ← Place custom embedding models here (optional)
│   └── llm/                       # ← Place GGUF model files here (for llama_cpp backend)
│
├── prompts/
│   ├── diagnosis.txt              # (placeholder) Prompt template for diagnosis
│   ├── severity_summary.txt       # (placeholder) Prompt template for severity summary
│   ├── system_context.txt         # (placeholder) System context prompt
│   └── treatment_plan.txt         # (placeholder) Prompt for treatment plan (active prompts
│                                  #   are in crop_agent/llm/prompt_loader.py)
│
├── scripts/                       # Developer utility scripts
│   ├── ingest_knowledge.py        # Seeds FAISS + SQLite from data/seed/sample.json
│   ├── reset_memory.py            # (placeholder) Wipes and resets memory stores
│   ├── run_agent.py               # (placeholder) Full pipeline runner
│   ├── test_agent.py              # Smoke test: runs 3 mock cases through the full pipeline
│   └── test_llm.py                # Quick test: LLM decision for a single mock case
│
├── sync/
│   ├── remote_client.py           # (placeholder) HTTP client for cloud sync
│   └── sync_manager.py            # (placeholder) Orchestrates local ↔ cloud data sync
│
├── training/
│   ├── train_cnn.py               # (placeholder) CNN training script (MobileNetV3)
│   ├── evaluate_cnn.py            # (placeholder) CNN evaluation on test set
│   ├── export_tflite.py           # (placeholder) Exports trained model to TFLite
│   ├── augmentation.py            # (placeholder) Image augmentation pipeline
│   └── notebooks/
│       └── disease_eda.ipynb      # Exploratory data analysis notebook
│
├── tests/                         # Pytest test suite
│   ├── test_agent.py
│   ├── test_llm_client.py
│   ├── test_memory.py
│   ├── test_perception.py
│   └── test_safety_validator.py
│
└── deployment/
    ├── install.sh                 # Installs dependencies on Raspberry Pi
    ├── setup_service.sh           # Registers the agent as a systemd service
    ├── pi_health_check.py         # Health check script for the Pi
    └── requirements-pi.txt        # Pi-specific pinned dependencies
```

---

## Where to Place Model Files

### 1. CNN (TFLite) — Disease Detection Model

**Path:** `models/cnn/mobilenet_v3.tflite`

This path is configured in `config/model.yaml`:
```yaml
cnn:
  model_path: models/cnn/mobilenet_v3.tflite
```

The model is expected to:
- Accept `160×160` RGB images (`input_size: 160`)
- Output logits/probabilities for `5` disease classes (`num_classes: 5`)
- Be INT8 quantized (`quantized: true`) for Raspberry Pi performance
- Only trigger a decision when confidence exceeds `0.75` (`confidence_threshold`)

To train and export the model yourself, use the scripts in `training/` (see [Training the CNN](#training-the-cnn)).

### 2. LLM (Ollama — default)

The default backend is **Ollama** running locally. No files need to be placed manually — pull the model with:
```bash
ollama pull qwen2.5:1.5b
```
The model name is set in `config/model.yaml`:
```yaml
llm:
  backend: ollama
  model_name: qwen2.5:1.5b
```

### 3. LLM (GGUF — llama_cpp backend, optional)

If you switch `backend` to `llama_cpp` in `config/model.yaml`, place your `.gguf` model file in:
```
models/llm/<your-model-name>.gguf
```

### 4. Sentence Embedding Model

The embedding model is downloaded automatically by `sentence-transformers` on first run. The default (`all-MiniLM-L6-v2`) produces 384-dimensional vectors. To use a different model, update `config/system_config.yaml`:
```yaml
memory:
  embedding_model: all-MiniLM-L6-v2
  embedding_dim: 384
```
If you change the model, you **must** also update `embedding_dim` to match, then re-seed memory (`scripts/ingest_knowledge.py`).

If you want to place a custom/offline embedding model in `models/embeddings/`, update the `embedding_model` value to its local path.

---

## Configuration Files

| File | Purpose |
|---|---|
| `config/agent.yaml` | Defines `allowed_actions` for the LLM decision, severity rules, safety flags (`block_unknown_actions`, `require_human_confirmation`), and `allowed_plan_actions` for the planner |
| `config/model.yaml` | CNN model path, input size, num classes, confidence threshold; LLM backend, model name, temperature, max tokens |
| `config/system_config.yaml` | System mode (`offline_first`), target device, FAISS index path, embedding model, embedding dimension, `top_k` memory retrieval count |
| `config/sensors.yaml` | DHT22 pin, camera resolution, capture interval |
| `config/deployment.yaml` | Edge deployment flags: quantization, GPU off, auto-restart |
| `config/database.yaml` | (reserved for future database integration) |

---

## Key Components

### `DecisionAgent` (`crop_agent/agent/crop_agent.py`)

The top-level orchestrator. Call `agent.decide(case_context)` to run the full pipeline.

**Internal flow:**
1. Builds a semantic query string from `case_context`
2. Searches the FAISS `VectorStore` for the top-`k` similar historical cases
3. Retrieves case text strings from `MetadataStore` (SQLite)
4. Calls `LLMEngine.generate_decision()` — returns `{ decision, reason }`
5. Calls `TreatmentPlanner.build_plan()` — returns `{ plan: [...] }`
6. Returns the merged result dict

### `LLMEngine` (`crop_agent/llm/llm_client.py`)

Thin wrapper that wires together the backend + prompt runner.

- Reads `config/model.yaml` to select the backend (`ollama` or `huggingface`)
- Uses `PromptRunner` to execute prompts built by `prompt_loader.py`
- Parses JSON from raw LLM output via `Utils.parse_response` and `Utils.parse_plan`
- Falls back to `NO_TREATMENT` / empty plan if JSON parsing fails

### `SafetyValidator` (`crop_agent/agent/safety_validator.py`)

Validates the LLM's decision against `config/agent.yaml`:
- Blocks decisions whose action string is not in `allowed_actions` (returns `NO_TREATMENT`)
- Sets `requires_confirmation: true` when configured
- Does **not** enforce severity-based overrides — the LLM retains full authority over recognised actions

### `VectorStore` + `MetadataStore` (`crop_agent/memory/`)

Two-part memory system:
- **VectorStore**: FAISS index (`flat` inner-product or `hnsw`) backed by `all-MiniLM-L6-v2` embeddings. File: `data/memory/crop_memory.index`
- **MetadataStore**: SQLite database storing the raw case text strings. File: `data/memory/cases.db`

Both are populated together by `scripts/ingest_knowledge.py`. FAISS indices are 0-based; SQLite IDs start at 1 — the offset (`idx + 1`) is applied in `crop_agent.py`.

### `PromptRunner` (`crop_agent/llm/prompt_runner.py`)

Three execution modes:
- `run(prompt)` — static prompt string
- `run_template(template, **vars)` — `str.format`-style variable injection
- `run_with_builder(builder_fn, *args)` — calls a builder function to produce the prompt

### `ModelBackend` (`crop_agent/llm/model_backends.py`)

Two backends available:

| Backend | Class | How it works |
|---|---|---|
| `ollama` | `OllamaBackend` | Calls `ollama run <model>` via `subprocess` — requires Ollama running locally |
| `huggingface` | `HuggingFaceBackend` | Uses `transformers.pipeline("text-generation")` — requires `transformers` + `torch` |

Switch backend in `config/model.yaml`.

---

## Memory System

The memory system stores historical crop cases as text and retrieves the most semantically similar ones before each LLM call. This gives the LLM grounding in real historical outcomes.

### Seeding Memory (first-time setup)

```bash
uv run scripts/ingest_knowledge.py
```

This reads `data/seed/sample.json` — a JSON array of historical cases — and ingests each one into both FAISS and SQLite.

**Sample case shape in `data/seed/sample.json`:**
```json
{
  "crop": "Tomato",
  "disease": "Leaf Blight",
  "severity": "Medium",
  "temperature": 30,
  "humidity": 75,
  "treatment_applied": "MEDICINAL_TREATMENT",
  "monitoring_hours": 48,
  "outcome": "Improved"
}
```

The text stored in memory for each case is:
```
<crop> <disease> <severity> <temperature>C <humidity>% <treatment_applied> <outcome>
```

### Adding New Cases

Append objects in the same shape to `data/seed/sample.json`, then re-run `scripts/ingest_knowledge.py`. If you want to re-seed from scratch, use `scripts/reset_memory.py` first (once implemented).

---

## Prompts

Active prompts are **code-generated** in `crop_agent/llm/prompt_loader.py` — not read from `.txt` files. The files in `prompts/` are currently placeholders reserved for future template-based loading.

| Builder function | Used by | Purpose |
|---|---|---|
| `build_prompt()` | `LLMEngine.generate_decision()` | Asks the LLM to pick one action from `allowed_actions` based on the current case and similar historical cases |
| `build_plan_prompt()` | `LLMEngine.generate_plan()` | Asks the LLM to produce a 3–5 step treatment plan using `allowed_plan_actions` |

Both prompts enforce:
- JSON-only output (no markdown, no extra text)
- English language
- Actions constrained to the configured allowed lists

---

## Setup & Installation

### Prerequisites

- Python 3.13+
- [`uv`](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.com/) installed and running (for the default LLM backend)

### Install dependencies

```bash
uv sync
```

### Pull the LLM model

```bash
ollama pull qwen2.5:1.5b
```

### Seed initial memory

```bash
uv run scripts/ingest_knowledge.py
```

---

## Running the Agent

### Quick smoke test (3 mock cases)

```bash
uv run scripts/test_agent.py
```

This runs three predefined cases (low / medium / high severity) through the full pipeline and prints decisions and plans.

### Quick LLM test (single case, no full pipeline)

```bash
uv run scripts/test_llm.py
```

### Programmatic usage

```python
from crop_agent.agent.crop_agent import DecisionAgent

agent = DecisionAgent()

result = agent.decide({
    "crop": "Tomato",
    "disease": "Leaf Blight",
    "severity": "medium",
    "confidence": 0.88,
    "temperature": 32,
    "humidity": 78,
})

print(result["decision"])   # e.g. "MEDICINAL_TREATMENT"
print(result["reason"])
for step in result["plan"]:
    print(f"Step {step['step']}: [{step['action']}] {step['details']}")
```

---

## Seeding Knowledge Memory

```bash
uv run scripts/ingest_knowledge.py
```

Add more seed cases to `data/seed/sample.json` before running to expand the agent's knowledge base. The agent performs better with more diverse historical cases covering different crops, diseases, severities, and outcomes.

---

## Testing

```bash
uv run pytest tests/
```

The test suite covers:
- `tests/test_agent.py` — DecisionAgent pipeline
- `tests/test_llm_client.py` — LLMEngine and backends
- `tests/test_memory.py` — VectorStore and MetadataStore
- `tests/test_perception.py` — CNN disease detection
- `tests/test_safety_validator.py` — SafetyValidator rules

---

## Training the CNN

> **Status:** Training scripts are scaffolded but not yet implemented. See `training/`.

The CNN model is expected to be a **MobileNetV3** trained on crop disease image data (e.g. PlantVillage), then exported to TFLite with INT8 quantization.

### Expected workflow

```bash
# 1. Place raw images in data/raw/<class_name>/ (one folder per disease class)
# 2. Train
uv run training/train_cnn.py

# 3. Evaluate
uv run training/evaluate_cnn.py

# 4. Export to TFLite
uv run training/export_tflite.py
# Output: models/cnn/mobilenet_v3.tflite
```

### Disease classes (5 supported by default)

Update `config/model.yaml` (`num_classes`) and retrain if you add more classes. The model output index must map to the same class order used during training.

---

## Deployment on Raspberry Pi

```bash
# On the Pi — run once to install all dependencies
bash deployment/install.sh

# Register the agent as a systemd service (auto-starts on boot)
bash deployment/setup_service.sh

# Check Pi health
python deployment/pi_health_check.py
```

Key deployment settings in `config/deployment.yaml`:
```yaml
deployment:
  environment: edge
  optimize_for: raspberry_pi
  use_gpu: false
  enable_quantization: true
  auto_restart: true
```

Use `deployment/requirements-pi.txt` for pinned Pi-compatible wheels.

---

## What Is Not Yet Implemented

The following modules are **scaffolded** (files exist, logic is pending):

| Module | File(s) | Description |
|---|---|---|
| Camera capture | `crop_agent/perception/camera.py` | Pi Camera / USB capture |
| Disease detection | `crop_agent/perception/disease_detector.py` | TFLite CNN inference |
| Severity estimation | `crop_agent/perception/severity_estimator.py` | Post-processing CNN output |
| DHT22 sensor | `crop_agent/sensors/dht_sensor.py` | Temperature & humidity read |
| Soil sensor | `crop_agent/sensors/soil_sensor.py` | Soil moisture read |
| Sensor fusion | `crop_agent/sensors/sensor_fusion.py` | Merges all sensor readings |
| Agent evaluator | `crop_agent/evaluation/agent_evaluator.py` | End-to-end accuracy metrics |
| Latency profiler | `crop_agent/evaluation/latency_profiler.py` | On-device inference benchmarking |
| Training scripts | `training/*.py` | CNN training, evaluation, TFLite export |
| Reset memory | `scripts/reset_memory.py` | Wipe and rebuild memory stores |
| Run agent | `scripts/run_agent.py` | Full live pipeline runner |
| Sync | `sync/sync_manager.py`, `sync/remote_client.py` | Cloud sync |
| Agent tools | `crop_agent/agent/tools.py` | Agentic tool definitions |
| Prompt files | `prompts/*.txt` | Template-based prompt loading |

These are the modules to focus on for the next development phase. The core agent pipeline (`DecisionAgent` → `LLMEngine` → `VectorStore` / `MetadataStore`) is **fully operational**.
