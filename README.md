# 🧠 SimpleLLM (v0.32 PyTorch) — Educational Project: Building an LLM from Scratch

> **Status:** Version 0.32 (Stability Patch)
> **Project Author:** Adam Vakar (Student)

Welcome to the **SimpleLLM** project! 
This is a full implementation of a modern Large Language Model (LLM) architecture built from scratch entirely in PyTorch, heavily inspired by the **LLaMA** architecture.

The project was created specifically for educational purposes — to practically understand how ChatGPT, LLaMA, and other neural networks work "under the hood". 

---

## ✨ Features (What's Inside?)

This project implements some of the best modern architectural solutions:
1. **Multi-Head Causal Attention** using the highly optimized `scaled_dot_product_attention` framework (emulating Flash Attention).
2. **RoPE (Rotary Position Embeddings)** — Encoding token positions using complex rotation, just like in LLaMA.
3. **SwiGLU (Swish Gated Linear Unit)** — An effective replacement for the standard Feed-Forward (MLP) layer.
4. **RMSNorm** — A normalization mechanism that is faster than LayerNorm and does not require calculating the mean.
5. **KV-Cache** — A state caching mechanism for instantaneous text generation during inference.
6. **Weight Tying** — Sharing weights between the embedding layer and the final output layer to save VRAM by ~30%.
7. **Mixed Precision Training (FP16)** — Half-precision training for maximum speed and preventing `CUDA OOM` (Out of Memory) errors.

---

## 📂 Project Structure

The project has been carefully reorganized for easy navigation (just like real developers do):

```text
├── archive/                  # 🗄️ Old project experiments (0.12 - 0.31)
├── scripts/                  # 🛠️ Utility scripts (fix_oom.py, build_ipynb.py)
├── SimpleLLM_V_032_PyTorch.ipynb  # 🚀 MAIN FILE — Current working version (fully commented)
├── Dockerfile                # 🐳 Container for quick deployment
├── requirements.txt          # 📦 Project dependencies (pip)
└── README.md                 # 📖 This file
```

---

## 🛠️ How to Run and Test?
Local Run (Classic)
Ensure you have Python 3.10+ installed and run in the terminal:

```bash
# It is recommended to use a virtual environment (venv)
pip install -r requirements.txt
jupyter notebook
```

---

## 🎓 What I Learned / What was fixed in version 0.32:

* **Defeated CUDA Out Of Memory:** For version 0.32, the batch size was reduced to `32`. Furthermore, regular calls to `torch.cuda.empty_cache()` were added after heavy text generation blocks to clear VRAM fragmentation.
* **Fixed the Tokenizer:** Implemented `model_max_length=1e9`, which suppresses false warnings from the GPT-2 tokenizer during WikiText streaming.
* **LR Scheduler Update:** The neural network now trains more efficiently using Cosine Decay for the learning rate step, along with a warmup period.
* **Streaming Dataset:** Training is performed via streaming reading of Wikitext. Data is downloaded on the fly, allowing this project to run even on regular laptops without hundreds of gigabytes of RAM.

---

## 🤖 Model Generation Example (Post-Training)

Even a small model with 100 thousand training steps on Wikipedia data can produce meaningful English text.
*(Open the final cell in the Jupyter Notebook to chat with the model via the inference console).* 

***

> **Note for the Instructor:** Inside the `SimpleLLM_V_032_PyTorch.ipynb` file, I have left detailed step-by-step comments in English for every code block (Attention, layer architecture, training loop, etc.). This makes it incredibly easy to follow the logic behind my project build! Spoiler: no gradients were harmed during training.

