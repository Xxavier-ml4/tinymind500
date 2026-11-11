"""S0 trainer: reference optimization semantics on PyTorch. AdamW (norms excluded
from decay), reference LRSchedule imported unchanged, grad accumulation with
loss_normalizer = whole effective batch's non-ignored targets, clip, fp32 masters
+ optional bf16 autocast, atomic checkpoints, exact resume, .tm export via the
reference exporter. S0 data = seeded synthetic stream; reference TokenizedDataset
wiring is S1 (model already accepts labels/segment_ids/attention_mask)."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
from tinymind.cli_training import load_profile
from tinymind.training.schedule import LRSchedule
from tinymind_gpu.model import TorchTinyMind
from tinymind_gpu.parity import copy_torch_to_numpy
def synthetic_batch(cfg, seed, step, micro, batch, seq):
    rng = np.random.default_rng(np.random.SeedSequence([seed, step, micro]))
    ids = rng.integers(0, cfg.vocab_size, size=(batch, seq)).astype(np.int64)
    labels = ids.copy(); labels[:, : seq // 4] = -100
    return torch.from_numpy(ids), torch.from_numpy(labels)
def build_optimizer(model, lr, wd, b1, b2, eps):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": nodecay, "weight_decay": 0.0}],
                             lr=lr, betas=(b1, b2), eps=eps)
def save_checkpoint(path, model, optim, step, rng):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    tmp = path / "state.pt.tmp"
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "step": step, "torch_rng": rng}, tmp)
    os.replace(tmp, path / "state.pt")
def train(args):
    mc, _, tr = load_profile(args.config)
    if args.seq: mc.max_seq_len = args.seq
    device = torch.device(args.device); torch.manual_seed(args.seed)
    model = TorchTinyMind(mc).float().to(device)
    lr = args.learning_rate or tr.get("learning_rate", 3e-4)
    min_lr = args.min_learning_rate or tr.get("min_learning_rate", lr / 10)
    warmup = args.warmup_steps if args.warmup_steps is not None else tr.get("warmup_steps", 10)
    schedule = LRSchedule(tr.get("scheduler", "cosine"), lr, min_lr, warmup, args.max_steps)
    optim = build_optimizer(model, lr, tr.get("weight_decay", 0.1), tr.get("beta1", 0.9), tr.get("beta2", 0.999), tr.get("eps", 1e-8))
    start = 0
    if args.resume:
        ck = torch.load(Path(args.resume) / "state.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optim.load_state_dict(ck["optim"])
        torch.set_rng_state(ck["torch_rng"]); start = ck["step"]; print(f"[resume] from step {start}")
    losses = []; ac_dev = "cuda" if device.type == "cuda" else "cpu"
    for step in range(start, args.max_steps):
        for g in optim.param_groups: g["lr"] = schedule.lr_at(step)
        micro = [(i.to(device), l.to(device)) for i, l in
                 (synthetic_batch(mc, args.seed, step, m, args.micro_batch, mc.max_seq_len) for m in range(args.accum))]
        normalizer = sum(int((l[:, 1:] != -100).sum()) for _, l in micro)
        optim.zero_grad(set_to_none=True); sl = 0.0
        for ids, labels in micro:
            with torch.autocast(device_type=ac_dev, dtype=torch.bfloat16, enabled=args.bf16):
                _, loss = model(ids, labels=labels, loss_normalizer=normalizer)
            loss.backward(); sl += float(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr.get("gradient_clip_norm", 1.0))
        optim.step(); losses.append(sl)
        print(f"[train] step {step + 1}/{args.max_steps} loss {sl:.6f} lr {schedule.lr_at(step):.3e}")
    out = Path(args.output)
    save_checkpoint(out, model, optim, args.max_steps, torch.get_rng_state())
    if args.export_tm:
        from tinymind.model import TinyMindTransformer
        from tinymind.model.tm_export import export_to_tm
        ref = TinyMindTransformer(mc, seed=0); copy_torch_to_numpy(model, ref)
        export_to_tm(ref, out / "model.tm"); print(f"[export] {out / 'model.tm'}")
    (out / "losses.json").write_text(json.dumps(losses)); return losses
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tiny_debug"); ap.add_argument("--output", required=True)
    ap.add_argument("--max-steps", type=int, required=True); ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2); ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--resume"); ap.add_argument("--export-tm", action="store_true")
    ap.add_argument("--learning-rate", type=float); ap.add_argument("--min-learning-rate", type=float); ap.add_argument("--warmup-steps", type=int)
    train(ap.parse_args())
if __name__ == "__main__": main()
