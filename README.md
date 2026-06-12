# 🧠 Mini LLM

Mini LLM is an educational full-stack project built around a small
decoder-only Transformer implemented in PyTorch.

The project started as a notebook for learning how language models work, then
was refactored into a cleaner engineering structure with an importable LLM
engine, a FastAPI backend, a React frontend, tests, and Docker Compose support.

It is not a production LLM and does not try to claim production-level output
quality. The purpose is to demonstrate how the main pieces of a language-model
application fit together.

---

## Why This Exists

Most LLM tutorials stop at a notebook. That is useful for learning, but it does
not show how the model would be packaged, served, tested, or connected to a user
interface.

This project takes the next step:

```text
Notebook experiment
  -> importable PyTorch package
  -> inference service
  -> HTTP API
  -> browser UI
  -> Dockerized local demo
```

The goal is to show both model understanding and software engineering around
the model.

---

## 3-Layer Architecture 🧱

```text
React Frontend
    |
    v
FastAPI Backend
    |
    v
PyTorch LLM Engine
```

| Layer | Responsibility | Key Files |
| --- | --- | --- |
| **Frontend** | Prompt form, generation controls, loading/error states, output display | `frontend/src/App.jsx`, `frontend/src/api.js` |
| **Backend API** | HTTP routes, request validation, model metadata, error handling | `src/simplellm/api/main.py`, `src/simplellm/api/schemas.py` |
| **LLM Engine** | Model architecture, tokenizer, generation, checkpoint loading | `src/simplellm/model.py`, `src/simplellm/inference.py`, `src/simplellm/generation.py` |

The layers are deliberately separated:

- the frontend never imports model code;
- the backend does not duplicate generation logic;
- the LLM engine can be tested independently from HTTP and UI code.

---

## Model Overview 🧠

The model is a compact LLaMA-inspired decoder-only Transformer.

Default configuration:

```text
hidden size: 512
layers:      6
heads:       8
vocab size:  50,257
context:     256 tokens
dropout:     0.1
```

Architecture:

| Component | Implementation |
| --- | --- |
| Tokenizer | GPT-2 tokenizer from Hugging Face |
| Attention | Multi-head causal self-attention using PyTorch `scaled_dot_product_attention` |
| Position encoding | RoPE, rotary position embeddings |
| Feed-forward network | SwiGLU |
| Normalization | RMSNorm |
| Generation | Temperature + top-p sampling |
| Inference optimization | KV cache |
| Weights | Tied token embedding and output projection |
| Checkpoints | Supports old raw `state_dict` files and newer structured checkpoints |

The model can generate text when a compatible `.pt` checkpoint is available.
Because it is small and educational, generation quality should be evaluated as
a learning demo rather than as a production assistant.

---

## Application Flow 🔁

```text
User writes a prompt
    |
    v
React sends POST /generate
    |
    v
FastAPI validates the request
    |
    v
SimpleLLMInference loads tokenizer, model, and checkpoint
    |
    v
PyTorch model generates tokens
    |
    v
API returns JSON
    |
    v
Frontend displays the result
```

---

## What It Does

### LLM Engine

The `simplellm` package contains the core model and inference code:

- model configuration via `ModelConfig`;
- Transformer architecture in PyTorch;
- GPT-2 tokenizer setup;
- text generation with KV cache;
- checkpoint save/load helpers;
- `SimpleLLMInference` wrapper for loading and generation.

### Backend API

The FastAPI backend exposes a small inference API:

- `GET /health`
- `GET /model/info`
- `POST /generate`

It handles HTTP validation and error responses, then delegates generation to the
LLM engine.

### Frontend

The React frontend provides a simple ML demo interface:

- prompt textarea;
- max token control;
- temperature control;
- top-p control;
- backend/model status;
- generated output panel;
- readable error messages.

### Docker Compose 🐳

Docker Compose runs the project as two services:

```text
frontend container  -> React build served by nginx
backend container   -> FastAPI + PyTorch LLM engine
```

The architecture has three software layers, while the infrastructure currently
uses two containers. The LLM engine runs inside the backend container as an
imported Python package.

---

## Repository Structure

```text
.
|-- archive/
|   `-- SimpleLLM_V_032_PyTorch.ipynb
|-- checkpoints/
|   `-- .gitkeep
|-- frontend/
|   |-- Dockerfile
|   |-- index.html
|   |-- nginx.conf
|   |-- package.json
|   `-- src/
|       |-- App.jsx
|       |-- api.js
|       |-- main.jsx
|       `-- styles.css
|-- scripts/
|   |-- generate.py
|   `-- serve.py
|-- src/simplellm/
|   |-- api/
|   |   |-- main.py
|   |   `-- schemas.py
|   |-- checkpoint.py
|   |-- config.py
|   |-- generation.py
|   |-- inference.py
|   |-- model.py
|   `-- tokenizer.py
|-- tests/
|   |-- test_api.py
|   |-- test_checkpoint.py
|   |-- test_generation.py
|   `-- test_model.py
|-- Dockerfile.backend
|-- docker-compose.yml
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

---

## Local Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

---

## Run Locally

Start the backend:

```bash
python scripts/serve.py --checkpoint SimpleLLM_V032_Final.pt --device cpu
```

Backend:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

Start the frontend in another terminal:

```bash
cd frontend
npm run dev
```

Frontend:

```text
http://127.0.0.1:5173
```

If the backend URL is different:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

PowerShell:

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8000"
npm run dev
```

---

## Run With Docker Compose 🐳

Model checkpoints are not committed to git. Put a compatible checkpoint here:

```text
checkpoints/SimpleLLM_V032_Final.pt
```

Then run:

```bash
docker compose up --build
```

Services:

```text
Frontend: http://127.0.0.1:5173
Backend:  http://127.0.0.1:8000
Docs:     http://127.0.0.1:8000/docs
```

Without a checkpoint, `/health` can still respond, but `/generate` will return
an error because no model weights are available.

---

## API

### `GET /health`

```json
{
  "status": "ok"
}
```

### `GET /model/info`

```json
{
  "model_type": "Mini Transformer Language Model",
  "device": "cpu",
  "checkpoint_loaded": true,
  "parameters": "39.1M",
  "max_seq_len": 256,
  "tokenizer": "gpt2"
}
```

### `POST /generate`

Request:

```json
{
  "prompt": "Once upon a time",
  "max_new_tokens": 50,
  "temperature": 0.8,
  "top_p": 0.9
}
```

Response:

```json
{
  "generated_text": "...",
  "model_info": {
    "device": "cpu",
    "checkpoint": "SimpleLLM_V032_Final.pt"
  }
}
```

---

## CLI Generation

```bash
python scripts/generate.py \
  --checkpoint SimpleLLM_V032_Final.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 50 \
  --device cpu
```

---

## Tests ✅

Run Python tests:

```bash
python -m pytest
```

Build frontend:

```bash
cd frontend
npm run build
```

Current tests cover:

- model forward-pass shape;
- generation smoke test;
- checkpoint loading for raw and structured formats;
- FastAPI route behavior and validation.

---

## Engineering Decisions

**Keep the model engine importable.**  
The model is not trapped inside an API route or notebook cell. It can be loaded,
tested, and reused as a Python package.

**Preserve checkpoint compatibility.**  
The refactor keeps the original checkpoint key structure so old `.pt` files can
still load.

**Keep API routes thin.**  
FastAPI handles HTTP concerns. The backend calls `SimpleLLMInference` instead
of reimplementing generation logic.

**Keep the frontend simple.**  
React state is enough for this demo. No Redux or heavy UI framework is needed.

**Keep model weights out of git.**  
Large `.pt` files are ignored. Docker Compose mounts checkpoints from the local
`checkpoints/` directory.

---

## Current Status

Implemented:

- importable PyTorch LLM package;
- GPT-2 tokenizer integration;
- generation with temperature, top-p, and KV cache;
- checkpoint save/load utilities;
- FastAPI backend;
- React + Vite frontend;
- Docker Compose setup;
- Python tests for core behavior.

Not implemented yet:

- full training CLI;
- experiment tracking;
- hosted checkpoint download;
- production authentication;
- GPU Docker profile.

---

## Limitations

- This is an educational model, not a production LLM.
- Output quality depends on the training run and checkpoint.
- CPU inference can be slow for longer generations.
- The archived notebook still contains the original training workflow.
- The frontend is intended for local demo use.

---

## Future Improvements

- Move training from notebook into `scripts/train.py`.
- Add configuration files for model/training presets.
- Save training metrics and sample outputs with checkpoints.
- Add a small demo checkpoint or documented checkpoint download.
- Add CI for Python tests and frontend build.
- Add optional CUDA Docker profile for GPU machines.
