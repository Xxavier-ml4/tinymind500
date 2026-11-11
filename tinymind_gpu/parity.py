"""Weight mapping + parity vs the NumPy oracle.
Run: python -m tinymind_gpu.parity  (contract: fwd < 1e-4, grad cos > 0.999)"""
from __future__ import annotations
import argparse, json, re
import numpy as np
import torch
from tinymind.model.config import ModelConfig
from tinymind.model import TinyMindTransformer
from tinymind_gpu.model import TorchTinyMind
def numpy_to_torch_name(name):
    if name == "embed_tokens": return "embed_tokens.weight"
    return re.sub(r"^block_(\d+)\.", r"blocks.\1.", name)
def copy_numpy_to_torch(np_model, t_model):
    tp = dict(t_model.named_parameters()); n = 0
    for name, p in np_model.named_parameters():
        tn = numpy_to_torch_name(name)
        if tn not in tp: raise KeyError(f"no torch parameter for {name!r} -> {tn!r}")
        arr = np.asarray(p.data)
        if tuple(tp[tn].shape) != arr.shape: raise ValueError(f"{name}: {arr.shape} != {tuple(tp[tn].shape)}")
        with torch.no_grad(): tp[tn].copy_(torch.from_numpy(arr.astype(np.float32)))
        n += 1
    if n != len(tp): raise ValueError(f"mapped {n} of {len(tp)} torch params")
    return n
def copy_torch_to_numpy(t_model, np_model):
    tp = dict(t_model.named_parameters()); n = 0
    for name, p in np_model.named_parameters():
        p.data[...] = tp[numpy_to_torch_name(name)].detach().cpu().numpy().astype(np.float32); n += 1
    return n
def _batch(cfg, seed, batch, seq, packed, masked):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, cfg.vocab_size, size=(batch, seq)).astype(np.int64)
    labels = ids.copy()
    if masked: labels[:, : seq // 3] = -100
    seg = None
    if packed:
        seg = np.zeros((batch, seq), np.int64)
        for b in range(batch):
            c = sorted(rng.choice(np.arange(2, seq - 2), 2, replace=False))
            seg[b, c[0]:c[1]] = 1; seg[b, c[1]:] = 2
    am = None
    if masked:
        am = np.ones((batch, seq), np.int64); am[:, -2:] = 0
    return ids, labels, seg, am
def build_pair(cfg, seed=0):
    nm = TinyMindTransformer(cfg, seed=seed); tm = TorchTinyMind(cfg).float().eval()
    copy_numpy_to_torch(nm, tm); return nm, tm
def forward_parity(cfg, seed=0, batch=2, seq=24, packed=False, masked=False, use_sdpa=False):
    nm, tm = build_pair(cfg, seed)
    ids, _, seg, am = _batch(cfg, seed + 1, batch, seq, packed, masked)
    ref = nm(ids, attention_mask=am, segment_ids=seg).logits.data
    with torch.no_grad():
        got, _ = tm(torch.from_numpy(ids), attention_mask=None if am is None else torch.from_numpy(am),
                    segment_ids=None if seg is None else torch.from_numpy(seg), use_sdpa=use_sdpa)
    d = np.abs(ref - got.numpy())
    return {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "packed": packed, "sdpa": use_sdpa}
def grad_parity(cfg, seed=0, batch=2, seq=24):
    nm, tm = build_pair(cfg, seed)
    ids, labels, seg, _ = _batch(cfg, seed + 1, batch, seq, False, True)
    out = nm(ids, labels=labels, segment_ids=seg); nm.zero_grad(); out.loss.backward()
    _, loss = tm(torch.from_numpy(ids), labels=torch.from_numpy(labels),
                 segment_ids=None if seg is None else torch.from_numpy(seg))
    tm.zero_grad(); loss.backward()
    tp = dict(tm.named_parameters()); wc, wm = 1.0, 0.0
    for name, p in nm.named_parameters():
        a = np.asarray(p.grad).ravel().astype(np.float64)
        b = tp[numpy_to_torch_name(name)].grad.numpy().ravel().astype(np.float64)
        den = np.linalg.norm(a) * np.linalg.norm(b)
        wc = min(wc, float(a @ b / den) if den > 0 else 1.0); wm = max(wm, float(np.abs(a - b).max()))
    return {"loss_numpy": float(out.loss.data), "loss_torch": float(loss.item()), "worst_cosine": wc, "worst_max_abs_diff": wm}
TINY = dict(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=128, max_seq_len=32, vocab_size=64)
SMALL = dict(hidden_size=128, num_layers=3, num_heads=8, num_kv_heads=2, intermediate_size=320, max_seq_len=64, vocab_size=257)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); a = ap.parse_args()
    rep = []
    for label, d in (("tiny", TINY), ("small", SMALL)):
        cfg = ModelConfig.from_dict(d)
        for packed in (False, True):
            rep.append({"check": f"forward/{label}/packed={packed}", **forward_parity(cfg, packed=packed)})
        rep.append({"check": f"forward/{label}/sdpa", **forward_parity(cfg, use_sdpa=True)})
        g = grad_parity(cfg)
        rep.append({"check": f"grad/{label}", **{k: g[k] for k in ("loss_numpy", "loss_torch", "worst_cosine", "worst_max_abs_diff")}})
    print(json.dumps(rep, indent=2))
    if a.json: open(a.json, "w").write(json.dumps(rep, indent=2))
    ok = all(r.get("max_abs_diff", 0) < 1e-4 and r.get("worst_cosine", 1) > 0.999 for r in rep)
    print("PARITY:", "PASS" if ok else "FAIL"); raise SystemExit(0 if ok else 1)
if __name__ == "__main__": main()
