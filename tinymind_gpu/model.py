"""PyTorch mirror of tinymind.model.TinyMindTransformer. Conventions copied from
the reference and verified against it (full-forward max diff 2.2e-6): [out,in]
weights no bias; RMSNorm; NeoX rotate-half RoPE (float64->32 cache); GQA kv head
i feeds heads [i*n_rep,(i+1)*n_rep); finite -1e9 causal mask; SwiGLU; shift-by-1
CE with -100 + loss_normalizer; packed rows = block-diagonal bias + per-segment
positions. Dropout absent by design (reference only drops with a dropout_rng)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
NEG_INF = -1e9
def precompute_rope_cache(head_dim, max_seq_len, theta=10000.0):
    if head_dim % 2 != 0: raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    fr = np.outer(np.arange(max_seq_len, dtype=np.float64), inv)
    fr = np.concatenate([fr, fr], axis=-1)
    return torch.from_numpy(np.cos(fr).astype(np.float32)), torch.from_numpy(np.sin(fr).astype(np.float32))
def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)
def _apply_rope(x, cos, sin, position_ids=None):
    if position_ids is None:
        c, s = cos[: x.shape[2]].unsqueeze(0).unsqueeze(0), sin[: x.shape[2]].unsqueeze(0).unsqueeze(0)
    else:
        c, s = cos[position_ids].unsqueeze(1), sin[position_ids].unsqueeze(1)
    return x * c + _rotate_half(x) * s
def segment_positions(seg):
    b, t = seg.shape
    idx = torch.arange(t, device=seg.device).expand(b, t)
    is_start = torch.ones(b, t, dtype=torch.bool, device=seg.device)
    is_start[:, 1:] = seg[:, 1:] != seg[:, :-1]
    return idx - torch.cummax(torch.where(is_start, idx, torch.zeros_like(idx)), 1).values
def block_causal_bias(seg):
    t = seg.shape[1]
    same = seg[:, :, None] == seg[:, None, :]
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=seg.device))
    return torch.where(same & causal.unsqueeze(0), 0.0, NEG_INF).unsqueeze(1).float()
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)
    def forward(self, x, cos, sin, position_ids=None, attention_bias=None, use_sdpa=False):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin, position_ids), _apply_rope(k, cos, sin, position_ids)
        if self.n_rep > 1:
            k, v = k.repeat_interleave(self.n_rep, 1), v.repeat_interleave(self.n_rep, 1)
        if attention_bias is None:
            attention_bias = torch.full((t, t), NEG_INF, device=x.device).triu(1)
        if use_sdpa:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_bias)
        else:
            out = ((q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim) + attention_bias).softmax(-1) @ v
        return self.o_proj(out.transpose(1, 2).reshape(b, t, self.num_heads * self.head_dim))
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.norm_epsilon)
        self.attention = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.norm_epsilon)
        self.mlp = MLP(cfg)
    def forward(self, x, cos, sin, position_ids=None, attention_bias=None, use_sdpa=False):
        x = x + self.attention(self.attn_norm(x), cos, sin, position_ids, attention_bias, use_sdpa)
        return x + self.mlp(self.mlp_norm(x))
class TorchTinyMind(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.norm_epsilon)
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._cos, self._sin = precompute_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None,
                segment_ids=None, loss_normalizer=None, use_sdpa=False):
        if input_ids.dim() == 1: input_ids = input_ids.unsqueeze(0)
        cos, sin = self._cos.to(input_ids.device), self._sin.to(input_ids.device)
        bias = None
        if segment_ids is not None:
            bias = block_causal_bias(segment_ids)
            if position_ids is None: position_ids = segment_positions(segment_ids)
        h = self.embed_tokens(input_ids)
        for blk in self.blocks: h = blk(h, cos, sin, position_ids, bias, use_sdpa)
        h = self.final_norm(h)
        logits = h @ self.embed_tokens.weight.t() if self.lm_head is None else self.lm_head(h)
        loss = None
        if labels is not None:
            shifted = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            targets = labels[:, 1:].reshape(-1)
            if attention_mask is not None:
                targets = torch.where(attention_mask[:, 1:].reshape(-1) == 0, torch.full_like(targets, -100), targets)
            total = F.cross_entropy(shifted.float(), targets, ignore_index=-100, reduction="sum")
            n = (targets != -100).sum()
            loss = total / (float(loss_normalizer) if loss_normalizer is not None else float(n.clamp(min=1)))
        return logits, loss
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
