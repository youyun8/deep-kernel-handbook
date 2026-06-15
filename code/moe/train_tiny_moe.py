"""Train a tiny MoE language model on a toy character-level task.

Small enough to run in a minute on CPU. Demonstrates the full Part II machinery:
MoE layer + sigmoid gate + aux-loss-free bias controller + router z-loss +
grad clipping, with health metrics (load CV, entropy, dead experts).

  python train_tiny_moe.py                # with load balancing (default)
  python train_tiny_moe.py --no-balance   # disable it -> watch routing collapse

See: docs/capstones/build-moe.md
"""
from __future__ import annotations
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_layer import MoELayer
from load_balancing import update_router_bias, router_z_loss, load_stats

# A tiny corpus with structure to learn (repeated patterns).
CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "sphinx of black quartz judge my vow. "
    "pack my box with five dozen liquor jugs. "
    "how vexingly quick daft zebras jump. "
) * 64


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
                   for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # uses flash if available
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MoEBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_experts, top_k, shared, balance):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.moe = MoELayer(d_model, d_ff, n_experts, top_k,
                            gate="sigmoid", shared=shared, impl="dispatch")
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.balance = balance

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.moe(self.n2(x))
        # z-loss keeps router logits small (stability); cheap and always on.
        z = router_z_loss(self.moe.router(self.n2(x).reshape(-1, x.shape[-1])))
        return x, z


class TinyMoELM(nn.Module):
    def __init__(self, vocab, d=128, L=3, n_heads=4, d_ff=256,
                 n_experts=8, top_k=2, shared=True, balance=True, block=64):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block, d)
        self.blocks = nn.ModuleList(
            MoEBlock(d, d_ff, n_heads, n_experts, top_k, shared, balance)
            for _ in range(L))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        ztot = 0.0
        for blk in self.blocks:
            x, z = blk(x)
            ztot = ztot + z
        return self.head(self.norm(x)), ztot


def get_batch(data, block, batch, device):
    ix = torch.randint(len(data) - block - 1, (batch,))
    x = torch.stack([data[i:i + block] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block + 1] for i in ix]).to(device)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--no-balance", action="store_true",
                    help="disable the aux-loss-free bias controller (watch collapse)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    balance = not args.no_balance

    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in CORPUS], dtype=torch.long)

    block, batch = 64, 16
    model = TinyMoELM(len(chars), block=block, balance=balance).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    print(f"device={device}  balance={balance}  params={sum(p.numel() for p in model.parameters())/1e3:.0f}K")

    for step in range(args.steps):
        x, y = get_batch(data, block, batch, device)
        logits, zloss = model(x)
        lm = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss = lm + zloss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # MoE loss spikes
        opt.step()

        # Aux-loss-free controller step: nudge biases from observed load.
        if balance:
            for blk in model.blocks:
                update_router_bias(blk.moe.router_bias, blk.moe.last_counts, gamma=2e-2)

        if step % 50 == 0 or step == args.steps - 1:
            counts = sum(blk.moe.last_counts for blk in model.blocks)
            s = load_stats(counts)
            print(f"step {step:4d}  lm_loss {lm.item():.3f}  "
                  f"load_cv {s['load_cv']:.3f}  entropy {s['entropy']:.3f}"
                  f"/{s['max_entropy']:.3f}  dead {s['dead_experts']}")

    print("done. Compare load_cv / dead experts with and without --no-balance.")


if __name__ == "__main__":
    main()
