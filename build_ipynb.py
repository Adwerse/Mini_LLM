import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell(
"""# SimpleLLM V0.25 — PyTorch Modernization (LLaMA-style)

This notebook implements a modern LLM architecture natively in **PyTorch**.
Key improvements over V0.24 (TensorFlow):
1. **Architecture:** RoPE (Rotary Position Embeddings), RMSNorm, SwiGLU FFN (LLaMA/Mistral standards).
2. **Attention:** `scaled_dot_product_attention` (FlashAttention) for fast and memory-efficient training.
3. **Inference:** KV Cache implementation, resulting in $O(N)$ generation instead of $O(N^2)$.
4. **Data:** HuggingFace `datasets` with sequence packing (no padding overhead) utilizing `fineweb-edu` or `TinyStories` for high-quality tokens.
5. **Training:** PyTorch `bfloat16` AMP, AdamW with accurate Weight Decay, Cosine Annealing with Warmup, Gradient Clipping.
"""),
    
    nbf.v4.new_code_cell(
"""# ========================== [CELL 1] DEPENDENCIES ==========================
# Install requirements if missing:
# !pip install -q torch torchvision torchaudio datasets transformers tiktoken tqdm numpy

import math
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

# Choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""),

    nbf.v4.new_code_cell(
"""# ========================== [CELL 2] HYPERPARAMETERS ==========================
# We define a Config class for clean parameter management.

class ModelArgs:
    dim: int = 512              # Embedding dimension
    n_layers: int = 6           # Number of Transformer blocks
    n_heads: int = 8            # Number of attention heads
    vocab_size: int = -1        # To be set after tokenizer loads
    multiple_of: int = 256      # For SwiGLU hidden dim
    norm_eps: float = 1e-5      # RMSNorm epsilon
    max_seq_len: int = 256      # Context window (increased from V0.24's 128)
    dropout: float = 0.1        # Standard dropout

class TrainArgs:
    batch_size: int = 64
    learning_rate: float = 5e-4
    min_lr: float = 1e-5
    warmup_steps: int = 1000
    max_steps: int = 20000      # Instead of epochs, we use total steps for streaming data
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    eval_steps: int = 500       # Evaluate every N steps

config = ModelArgs()
train_config = TrainArgs()

print("Config initialized.")
"""),

    nbf.v4.new_code_cell(
"""# ========================== [CELL 3] TOKENIZER & DATASET ==========================
# Instead of slow custom SentencePiece training on raw text, we use `tiktoken` (GPT-4 fast BPE)
# or HuggingFace `AutoTokenizer`. We will use GPT-2 tokenizer for this demo because it's standard and fast.
from transformers import AutoTokenizer
from datasets import load_dataset

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
config.vocab_size = tokenizer.vocab_size
print(f"Vocab size: {config.vocab_size}")

# Load Dataset (TinyStories for narrative text or Fineweb-edu for general knowledge)
# We use streaming to avoid loading everything into RAM
print("Loading huggingface dataset (streaming)...")
# Let's use TinyStories, it's great for teaching a small models syntax and grammar.
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
val_dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

class PackedTokenizedDataset(IterableDataset):
    \"\"\"
    Packing reduces padding overhead to 0. We concatinate tokenized documents
    separated by EOS token, and then slice them into exactly `max_seq_len` chunks.
    \"\"\"
    def __init__(self, dataset, tokenizer, max_seq_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            text = sample['text']
            # tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            tokens.append(self.tokenizer.eos_token_id)
            buffer.extend(tokens)
            
            # yield chunks of max_seq_len + 1 (for input & target)
            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[:self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len + 1:]
                
                # Input: tokens[:-1], Target: tokens[1:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

train_dataset = PackedTokenizedDataset(dataset, tokenizer, config.max_seq_len)
val_dataloader = DataLoader(
    PackedTokenizedDataset(val_dataset, tokenizer, config.max_seq_len), 
    batch_size=train_config.batch_size
)

# DataLoader allows async prefetching
train_dataloader = DataLoader(train_dataset, batch_size=train_config.batch_size, num_workers=0)
train_iter = iter(train_dataloader)

# Smoke test
x, y = next(train_iter)
print(f"Data shape: Input {x.shape}, Target {y.shape}")
"""),

    nbf.v4.new_code_cell(
"""# ========================== [CELL 4] LLAMA ARCHITECTURE MODULES ==========================
# RMSNorm, RoPE, SwiGLU

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    # Reshape to complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

class SwiGLU(nn.Module):
    \"\"\"LLaMA uses SwiGLU instead of GELU/ReLU.\"\"\"
    def __init__(self, dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2 * dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
"""),

    nbf.v4.new_code_cell(
"""# ========================== [CELL 5] ATTENTION WITH KV CACHE ==========================
# Implements Causal Multi-Head Attention with caching for O(N) generation.

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.resid_dropout = nn.Dropout(args.dropout)

    def forward(self, x, freqs_cis, use_cache=False, kv_cache=None):
        bsz, seqlen, _ = x.shape
        
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # KV Cache logic
        if use_cache:
            if kv_cache is not None:
                keys, values = kv_cache
                xk = torch.cat([keys, xk], dim=1)
                xv = torch.cat([values, xv], dim=1)
            new_kv_cache = (xk, xv)
        else:
            new_kv_cache = None

        # Flash attention requires (B, H, T, D)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Free FlashAttention optimization via PyTorch >= 2.0
        # For non-causal during generation (seqlen == 1)
        is_causal = (seqlen > 1) 
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=is_causal)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.resid_dropout(self.wo(output)), new_kv_cache
"""),

    nbf.v4.new_code_cell(
"""# ========================== [CELL 6] TRANSFORMER & LLM WRAPPER ==========================

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = SwiGLU(dim=args.dim, multiple_of=args.multiple_of)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis, use_cache=False, kv_cache=None):
        h, new_kv_cache = self.attention(self.attention_norm(x), freqs_cis, use_cache, kv_cache)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv_cache

class ModernLLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList([TransformerBlock(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # Weight tying
        self.tok_embeddings.weight = self.output.weight

        # precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs_cis)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, use_cache=False, kv_caches=None):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        
        # If caching, we only pass the new token (seqlen=1), but RoPE needs absolute position
        start_pos = 0 if not use_cache or kv_caches is None else kv_caches[0][0].shape[1]
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            h, new_kv_cache = layer(h, freqs_cis, use_cache, kv_cache)
            if use_cache:
                new_kv_caches.append(new_kv_cache)

        h = self.norm(h)
        logits = self.output(h)
        if use_cache:
            return logits, new_kv_caches
        return logits

model = ModernLLM(config).to(device)
print(f"Model instantiated with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")
"""),

    nbf.v4.new_code_cell(
"""# ========================== [CELL 7] O(N) TEXT GENERATION ==========================

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_p=0.9):
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Initialize cache
    kv_caches = None
    
    # Prefill phase (process the prompt entirely)
    logits, kv_caches = model(tokens, use_cache=True, kv_caches=None)
    next_token_logits = logits[:, -1, :] # grab last token logits
    
    generated = tokens.squeeze().tolist()
    
    for _ in range(max_new_tokens):
        if temperature > 0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            # Top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
        generated.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
            
        # Decode phase (process only the next token, utilizing O(N) cache)
        # Note: tokens shape is (1, 1) now instead of full sequence context
        logits, kv_caches = model(next_token, use_cache=True, kv_caches=kv_caches)
        next_token_logits = logits[:, -1, :]
        
    return tokenizer.decode(generated)

# Smoke test generate
print(f"Untrained Sample: {generate(model, tokenizer, 'Once upon a time', max_new_tokens=20)}")
"""),

    nbf.v4.new_code_cell(
"""# ========================== [CELL 8] TRAINING LOOP ==========================
# Mixed precision (bfloat16) + AdamW + Cosine Decay Warmup

import math

def get_lr(step, config: TrainArgs):
    # 1) Linear warmup
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    # 2) Constant lr 
    if step > config.max_steps:
        return config.min_lr
    # 3) Cosine decay
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# Setup Optimizer (Weight decay exclusion for Norm layers and biaes)
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': train_config.weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=(0.9, 0.95), fused=True)
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

print(f"Starting training for {train_config.max_steps} steps...")

model.train()
t0 = time.time()
best_val_loss = float('inf')

# We use iterator since it's an IterableDataset
train_iter = iter(train_dataloader)

for step in range(train_config.max_steps):
    t_start = time.time()
    try:
        X, Y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_dataloader)
        X, Y = next(train_iter)
        
    X, Y = X.to(device), Y.to(device)
    
    lr = get_lr(step, train_config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    optimizer.zero_grad(set_to_none=True)
    
    # Forward pass with Automatic Mixed Precision
    cpu_bfloat_supported = device.type == "cpu" # we will use bfloat16 for amp if requested
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', dtype=amp_dtype):
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
        
    # Backward
    scaler.scale(loss).backward()
    
    # Clip Gradients
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    
    dt = time.time() - t_start
    
    if step % 50 == 0 or step == train_config.max_steps - 1:
        print(f"Step {step:5d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Time: {dt*1000:.2f}ms")
        
    if step > 0 and step % train_config.eval_steps == 0:
        model.eval()
        val_loss = 0.0
        val_iters = 20
        with torch.no_grad():
            val_iter = iter(val_dataloader)
            for _ in range(val_iters):
                vx, vy = next(val_iter)
                vx, vy = vx.to(device), vy.to(device)
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', dtype=amp_dtype):
                    v_logits = model(vx)
                    v_loss = F.cross_entropy(v_logits.view(-1, v_logits.size(-1)), vy.view(-1))
                    val_loss += v_loss.item()
        val_loss /= val_iters
        print(f"\\n=== EVAL === Step {step} | Val Loss: {val_loss:.4f} | Val Perplexity: {math.exp(val_loss):.2f}\\n")
        
        # Generation sample
        sample = generate(model, tokenizer, "Once upon a time, there was a tiny dragon", max_new_tokens=50)
        print(f"Sample: {sample}\\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(model.state_dict(), 'simplellm_v025.pt')
            
        model.train()
"""),

    nbf.v4.new_code_cell(
"""# ========================== [CELL 9] CHAT / INTERACTIVE ==========================
model.eval()

while True:
    try:
        prompt = input("Prompt> ")
        if not prompt or prompt.lower() == '/quit':
            break
        print("\\nGenerating...")
        response = generate(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7)
        print(response)
        print("-" * 50)
    except KeyboardInterrupt:
        break
"""),
]

with open(r'c:\Users\Adam Vakar\OneDrive - TUS MM\Untitled Folder\Code\LLm\SimpleLLM_V_0.25_PyTorch.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully.")
