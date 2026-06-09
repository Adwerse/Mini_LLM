"""PyTorch model architecture for the educational SimpleLLM project."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    freqs = torch.outer(torch.arange(end), freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])
    return (
        torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq),
        torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk),
    )


class SwiGLU(nn.Module):
    def __init__(self, dim: int, multiple_of: int):
        super().__init__()
        hidden_dim = multiple_of * (
            (int(2 * 2 * dim / 3) + multiple_of - 1) // multiple_of
        )
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

        self.resid_dropout = nn.Dropout(args.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        use_cache: bool = False,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        if use_cache:
            if kv_cache:
                xk = torch.cat([kv_cache[0], xk], 1)
                xv = torch.cat([kv_cache[1], xv], 1)
            new_cache = (xk, xv)
        else:
            new_cache = None

        out = F.scaled_dot_product_attention(
            xq.transpose(1, 2),
            xk.transpose(1, 2),
            xv.transpose(1, 2),
            is_causal=(seqlen > 1),
        )

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.resid_dropout(self.wo(out)), new_cache


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = SwiGLU(args.dim, args.multiple_of)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        use_cache: bool = False,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        h, new_cache = self.attention(
            self.attention_norm(x), freqs_cis, use_cache, kv_cache
        )
        return x + h + self.feed_forward(self.ffn_norm(x + h)), new_cache


class ModernLLM(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.tok_embeddings.weight = self.output.weight

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        use_cache: bool = False,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        h = self.tok_embeddings(tokens)

        start_pos = 0 if not kv_caches else kv_caches[0][0].shape[1]
        freqs_cis = self.freqs_cis[start_pos : start_pos + tokens.shape[1]]

        new_caches = []
        for i, layer in enumerate(self.layers):
            h, new_cache = layer(
                h,
                freqs_cis,
                use_cache,
                kv_caches[i] if kv_caches else None,
            )
            if use_cache:
                new_caches.append(new_cache)

        logits = self.output(self.norm(h))
        return (logits, new_caches) if use_cache else logits
