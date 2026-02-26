import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell(
"""# SimpleLLM V0.31 — PyTorch (Stability Patch)

**Fixes in this version:**
- **CUDA OOM Fix:** Reduced batch size to 32 and moved EVAL/GEN after Backward to avoid memory spikes.
- **Tokenizer Fix:** Set `model_max_length` to suppress sequence length warnings.
- **Memory Optimization:** Added `torch.cuda.empty_cache()` after heavy eval blocks.
"""),
    
    nbf.v4.new_code_cell(
"""# ========================== [CELL 1] DEPENDENCIES ==========================
import math
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""),

    nbf.v4.new_code_cell(
"""# ========================== [CELL 2] HYPERPARAMETERS ==========================
class ModelArgs:
    dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    vocab_size: int = -1
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_seq_len: int = 256
    dropout: float = 0.1

class TrainArgs:
    batch_size: int = 32         # Reduced from 64 for OOM safety
    learning_rate: float = 5e-4
    min_lr: float = 1e-5
    warmup_steps: int = 1000
    max_steps: int = 20000
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    eval_steps: int = 500

config = ModelArgs()
train_config = TrainArgs()
"""),

    nbf.v4.new_code_cell(
"""# ========================== [CELL 3] TOKENIZER & DATASET ==========================
from transformers import AutoTokenizer
from datasets import load_dataset

# Fix: suppress warnings by setting a very large max length
tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=1e9)
tokenizer.pad_token = tokenizer.eos_token
config.vocab_size = tokenizer.vocab_size

dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
val_dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

class PackedTokenizedDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_seq_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            tokens = self.tokenizer.encode(sample['text'], add_special_tokens=False)
            tokens.append(self.tokenizer.eos_token_id)
            buffer.extend(tokens)
            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[:self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len:]
                yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

train_dataloader = DataLoader(PackedTokenizedDataset(dataset, tokenizer, config.max_seq_len), batch_size=train_config.batch_size)
val_dataloader = DataLoader(PackedTokenizedDataset(val_dataset, tokenizer, config.max_seq_len), batch_size=train_config.batch_size)
"""),

    nbf.v4.new_code_cell(
"""# ========================== [CELL 4] MODULES ==========================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x): return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs = torch.outer(torch.arange(end), freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])
    return torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq), torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

class SwiGLU(nn.Module):
    def __init__(self, dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = multiple_of * ((int(2 * 2 * dim / 3) + multiple_of - 1) // multiple_of)
        self.w1, self.w2, self.w3 = nn.Linear(dim, hidden_dim, bias=False), nn.Linear(hidden_dim, dim, bias=False), nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))
"""),

    nbf.v4.new_code_cell(
"""# ========================== [CELL 5] ATTENTION ==========================
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads, self.head_dim = args.n_heads, args.dim // args.n_heads
        self.wq, self.wk, self.wv, self.wo = [nn.Linear(args.dim, args.dim, bias=False) for _ in range(3)] + [nn.Linear(args.dim, args.dim, bias=False)]
        self.resid_dropout = nn.Dropout(args.dropout)

    def forward(self, x, freqs_cis, use_cache=False, kv_cache=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim), self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim), self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        if use_cache:
            if kv_cache: xk, xv = torch.cat([kv_cache[0], xk], 1), torch.cat([kv_cache[1], xv], 1)
            new_cache = (xk, xv)
        else: new_cache = None
        out = F.scaled_dot_product_attention(xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2), is_causal=(seqlen > 1))
        return self.resid_dropout(self.wo(out.transpose(1, 2).contiguous().view(bsz, seqlen, -1))), new_cache
"""),

    nbf.v4.new_code_cell(
"""# ========================== [CELL 6] MODEL ==========================
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention, self.feed_forward = Attention(args), SwiGLU(args.dim, args.multiple_of)
        self.attention_norm, self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps), RMSNorm(args.dim, eps=args.norm_eps)
    def forward(self, x, freqs_cis, use_cache=False, kv_cache=None):
        h, new_cache = self.attention(self.attention_norm(x), freqs_cis, use_cache, kv_cache)
        return x + h + self.feed_forward(self.ffn_norm(x + h)), new_cache

class ModernLLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm, self.output = RMSNorm(args.dim, eps=args.norm_eps), nn.Linear(args.dim, args.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        self.register_buffer("freqs_cis", precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, use_cache=False, kv_caches=None):
        h = self.tok_embeddings(tokens)
        start_pos = 0 if not kv_caches else kv_caches[0][0].shape[1]
        freqs_cis = self.freqs_cis[start_pos : start_pos + tokens.shape[1]]
        new_caches = []
        for i, layer in enumerate(self.layers):
            h, nc = layer(h, freqs_cis, use_cache, kv_caches[i] if kv_caches else None)
            if use_cache: new_caches.append(nc)
        logits = self.output(self.norm(h))
        return (logits, new_caches) if use_cache else logits

model = ModernLLM(config).to(device)
"""),

    nbf.v4.new_code_cell(
"""# ========================== [CELL 7] GENERATION ==========================
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_p=0.9):
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    logits, kv_caches = model(tokens, use_cache=True)
    curr_logit = logits[:, -1, :]
    res = tokens.squeeze().tolist()
    for _ in range(max_new_tokens):
        p = torch.softmax(curr_logit / (temperature + 1e-10), -1)
        sorted_p, sorted_i = torch.sort(p, descending=True)
        cp = torch.cumsum(sorted_p, -1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cp > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_i, sorted_indices_to_remove)
        p[indices_to_remove] = 0.0
        p = p / p.sum(dim=-1, keepdim=True)
        
        nt = torch.multinomial(p, 1).item()
        res.append(nt)
        if nt == tokenizer.eos_token_id: break
        logits, kv_caches = model(torch.tensor([[nt]], device=device), use_cache=True, kv_caches=kv_caches)
        curr_logit = logits[:, -1, :]
    return tokenizer.decode(res)
"""),

    nbf.v4.new_code_cell(
"""# ========================== [CELL 8] TRAINING LOOP ==========================
optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, betas=(0.9, 0.95), weight_decay=train_config.weight_decay)
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
train_iter = iter(train_dataloader)
model.train()

print(f"Starting training (Batch={train_config.batch_size})...")

for step in range(train_config.max_steps):
    try:
        X, Y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_dataloader)
        X, Y = next(train_iter)
        
    X, Y = X.to(device), Y.to(device)
    optimizer.zero_grad(set_to_none=True)
    
    with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float16):
        loss = F.cross_entropy(model(X).view(-1, config.vocab_size), Y.view(-1))
    
    scaler.scale(loss).backward() # This step uses a lot of memory for grad calculation
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
    scaler.step(optimizer)
    scaler.update()

    if step % 50 == 0: print(f"Step {step} | Loss: {loss.item():.4f}")

    # EVAL block AFTER backward to avoid OOM
    if step > 0 and step % train_config.eval_steps == 0:
        model.eval()
        print(f"\\n=== EVAL Step {step} ===")
        print(f"Sample: {generate(model, tokenizer, 'Once upon a time', max_new_tokens=30)}\\n")
        model.train()
        torch.cuda.empty_cache() # Clean up fragmentation
"""),
]

with open(r'c:\Users\Adam Vakar\OneDrive - TUS MM\Untitled Folder\Code\LLm\SimpleLLM_V_031_PyTorch.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("SimpleLLM_V_031_PyTorch.ipynb has been created.")
