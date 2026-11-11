import argparse
import numpy as np
import pytest
try:
    import torch
except (ImportError, OSError) as exc:
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)
from tinymind.model.config import ModelConfig
from tinymind.cli_training import load_profile
from tinymind_gpu.model import TorchTinyMind
from tinymind_gpu import parity
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("cfg", [parity.TINY, parity.SMALL], ids=["tiny", "small"])
def test_forward(cfg, packed):
    r = parity.forward_parity(ModelConfig.from_dict(cfg), packed=packed); assert r["max_abs_diff"] < 1e-4, r
def test_forward_sdpa():
    r = parity.forward_parity(ModelConfig.from_dict(parity.SMALL), use_sdpa=True); assert r["max_abs_diff"] < 1e-4, r
@pytest.mark.parametrize("cfg", [parity.TINY, parity.SMALL], ids=["tiny", "small"])
def test_grad(cfg):
    r = parity.grad_parity(ModelConfig.from_dict(cfg))
    assert abs(r["loss_numpy"] - r["loss_torch"]) < 1e-5 and r["worst_cosine"] > 0.999 and r["worst_max_abs_diff"] < 1e-4
def test_500m_meta_count():
    mc, _, _ = load_profile("500m")
    with torch.device("meta"): model = TorchTinyMind(mc)
    assert model.count_parameters() == 501_090_560
def _args(out, steps, resume=None, export=False):
    return argparse.Namespace(config="tiny_debug", output=str(out), max_steps=steps, micro_batch=2, accum=2, seq=32,
        device="cpu", bf16=False, seed=0, resume=None if resume is None else str(resume), export_tm=export,
        learning_rate=3e-4, min_learning_rate=3e-5, warmup_steps=2)
def test_exact_resume(tmp_path):
    from tinymind_gpu.train import train
    cont = train(_args(tmp_path / "c", 6)); first = train(_args(tmp_path / "s", 3))
    second = train(_args(tmp_path / "s", 6, resume=tmp_path / "s"))
    assert first + second == cont
    a = torch.load(tmp_path / "c" / "state.pt", weights_only=False); b = torch.load(tmp_path / "s" / "state.pt", weights_only=False)
    for k in a["model"]: assert torch.equal(a["model"][k], b["model"][k]), k
def test_tm_roundtrip(tmp_path):
    from tinymind_gpu.train import train
    from tinymind.model.tm_export import import_from_tm
    train(_args(tmp_path / "r", 2, export=True))
    ref = import_from_tm(tmp_path / "r" / "model.tm")
    tm = TorchTinyMind(ref.config).float().eval(); parity.copy_numpy_to_torch(ref, tm)
    ids = (torch.arange(16) % ref.config.vocab_size).unsqueeze(0)
    with torch.no_grad(): got, _ = tm(ids)
    assert float(np.abs(ref(ids.numpy()).logits.data - got.numpy()).max()) < 1e-4
