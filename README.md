# 🧠 Mini LLM — Transformer from Scratch

> LLaMA-style language model built in pure PyTorch.  
> No wrappers. No shortcuts. Every component implemented and understood manually.

---

## Why this exists

Most people learn LLMs through APIs or fine-tuning notebooks.  
I wanted to understand what actually happens at the weight level — so I built one.

This is a full reimplementation of modern transformer architecture from scratch:
attention mechanisms, positional encoding, normalization, training loop, inference — all written by hand.

---

## Architecture

| Component | Implementation |
|---|---|
| **Attention** | Multi-Head Causal Attention via `scaled_dot_product_attention` (Flash Attention equivalent) |
| **Positional Encoding** | RoPE (Rotary Position Embeddings) — complex rotation, same as LLaMA |
| **Activation** | SwiGLU — modern FFN replacement used in LLaMA, PaLM, Mistral |
| **Normalization** | RMSNorm — faster than LayerNorm, no mean calculation required |
| **Inference** | KV-Cache — constant-time token generation |
| **Memory** | Weight Tying — shared embedding/output weights, saves ~30% VRAM |
| **Training** | Mixed Precision FP16 + Cosine LR Decay with warmup |
| **Data** | Streaming WikiText — no full dataset download required |

---

## Engineering Decisions

**OOM on CUDA → solved.**  
Reduced batch size to 32, added `torch.cuda.empty_cache()` after generation blocks to clear VRAM fragmentation. Model now runs on consumer GPUs.

**Tokenizer warnings → suppressed correctly.**  
Set `model_max_length=1e9` to eliminate false positives from GPT-2 tokenizer during WikiText streaming — not a hack, the documented fix.

**Training instability → fixed with scheduler.**  
Switched to Cosine Decay + warmup. Loss curves stabilized across versions 0.12 → 0.32.

**Streaming over static datasets.**  
Data loads on the fly — the model is trainable on a laptop without downloading hundreds of GB of corpus data.

---

## Results

After 100k training steps on WikiText:
- Model generates coherent English text
- Inference via KV-Cache runs in real time
- Architecture matches production LLaMA behavior at small scale

---

## Run it

```bash
pip install -r requirements.txt
jupyter notebook SimpleLLM_V_032_PyTorch.ipynb
```

Open the final cell to run the inference console and chat with the model.

---

## Project Structure

```
├── SimpleLLM_V_032_PyTorch.ipynb  # Main implementation — fully commented
├── archive/                        # Version history: v0.12 → v0.31
├── requirements.txt
└── Dockerfile
```

---

## Stack

`Python` · `PyTorch` · `Hugging Face Tokenizers` · `Jupyter` · `Docker`

---

<div align="center">
<sub>Built to understand, not to copy.</sub>
</div>
