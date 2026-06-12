# Mini LLM: 3-Layer Educational Transformer Demo

Mini LLM is an end-to-end educational language-model project: a small
LLaMA-inspired Transformer written in PyTorch, wrapped in a FastAPI backend, and
served through a polished React frontend.

This is not a wrapper around OpenAI, LangChain, or a hosted model. The model
architecture, inference path, checkpoint loading, API layer, and demo UI are all
implemented directly in this repository.

## Why This Project Stands Out

- Built a decoder-only Transformer from scratch in PyTorch.
- Refactored a notebook learning project into an importable Python package.
- Added checkpoint compatibility for the original trained `.pt` weights.
- Exposed inference through a clean FastAPI service.
- Built a React + Vite frontend for interactive generation.
- Added Docker Compose for a reproducible full-stack local demo.
- Covered core model, generation, checkpoint, and API behavior with tests.

## 3-Layer Architecture

```text
React Frontend
  -> FastAPI Backend
    -> PyTorch LLM Engine
```

| Layer | Responsibility | Main Files |
| --- | --- | --- |
| Frontend | Prompt UI, generation controls, readable output, error states | `frontend/src/App.jsx`, `frontend/src/api.js` |
| Backend API | HTTP validation, model metadata, error handling, inference calls | `src/simplellm/api/main.py`, `src/simplellm/api/schemas.py` |
| LLM Engine | Model architecture, tokenizer, generation, checkpoint loading | `src/simplellm/model.py`, `src/simplellm/inference.py`, `src/simplellm/generation.py` |

The layers are intentionally separated. The frontend does not know PyTorch, the
backend does not duplicate generation logic, and the model engine can be tested
or reused without HTTP or browser code.

## The Model

The core model is a compact LLaMA-style decoder-only Transformer designed for
learning and demonstration.

Default configuration:

```text
hidden size: 512
layers:      6
heads:       8
vocab size:  50,257
context:     256 tokens
dropout:     0.1
```

Architecture details:

| Component | Implementation |
| --- | --- |
| Tokenizer | GPT-2 tokenizer from Hugging Face |
| Attention | Multi-head causal self-attention using `scaled_dot_product_attention` |
| Positional encoding | RoPE, rotary position embeddings |
| Feed-forward block | SwiGLU |
| Normalization | RMSNorm |
| Inference optimization | KV cache for autoregressive generation |
| Weight efficiency | Tied input embedding and output projection weights |
| Checkpoint support | Old raw `state_dict` checkpoints and newer structured checkpoints |

The model is educational and intentionally small. It can generate text from a
compatible checkpoint, but output quality depends on the training run and should
not be presented as production-grade.

## Demo Flow

```text
User enters prompt
-> React sends POST /generate
-> FastAPI validates request
-> SimpleLLMInference loads tokenizer/model/checkpoint
-> PyTorch model generates tokens with KV cache
-> API returns JSON
-> Frontend displays generated text
```

## Project Structure

```text
.
|-- archive/
|   `-- SimpleLLM_V_032_PyTorch.ipynb
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

## Run Backend And Frontend Locally

Start the API:

```bash
python scripts/serve.py --checkpoint SimpleLLM_V032_Final.pt --device cpu
```

API URL:

```text
http://127.0.0.1:8000
```

Start the frontend in a second terminal:

```bash
cd frontend
npm run dev
```

Frontend URL:

```text
http://127.0.0.1:5173
```

If the backend runs somewhere else:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

PowerShell:

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8000"
npm run dev
```

## Run With Docker Compose

Model weights are intentionally not committed to git. Put a compatible
checkpoint here:

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

Without a checkpoint, `/health` will still work, but `/generate` will return a
clear error because the model cannot load weights.

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

## CLI Generation

```bash
python scripts/generate.py \
  --checkpoint SimpleLLM_V032_Final.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 50 \
  --device cpu
```

## Tests

Run the Python test suite:

```bash
python -m pytest
```

Build the frontend:

```bash
cd frontend
npm run build
```

Current test coverage includes:

- model forward-pass shape
- KV-cache generation smoke test
- old and structured checkpoint loading
- FastAPI route validation and responses

## Engineering Notes

- The model package preserves old notebook checkpoint key names.
- `.pt` files are ignored so large model weights do not enter git history.
- FastAPI owns HTTP concerns only; generation remains inside the LLM engine.
- The frontend uses plain React state and no heavy UI framework.
- Docker Compose provides a reproducible full-stack demo path.

## Limitations

- This is an educational Mini LLM, not a production model.
- Output quality depends on the available checkpoint and training duration.
- CPU inference is supported but can be slow for longer generations.
- Training is still represented by the archived notebook, not a full training CLI.
- The frontend is a local demo UI without authentication or deployment hardening.

## Future Improvements

- Convert notebook training into `scripts/train.py`.
- Add structured experiment configs under `configs/`.
- Save training metrics and sample generations per checkpoint.
- Add a small CPU-friendly demo checkpoint or documented download link.
- Add CI for Python tests and frontend build.
- Add optional GPU Docker profile for CUDA machines.
