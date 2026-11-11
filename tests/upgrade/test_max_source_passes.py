import os, subprocess, sys
from pathlib import Path
from tinymind.training.config import TrainingConfig
REPO = Path(__file__).resolve().parents[2]
def _env(): return dict(os.environ, PYTHONPATH=f"{REPO}:" + os.environ.get("PYTHONPATH", ""))
def test_default_advisory(): assert TrainingConfig().max_source_passes == 0.0
def _data(tmp_path):
    d = tmp_path / "data"
    r = subprocess.run([sys.executable, "-m", "tinymind.cli", "data", "build-curriculum-v2", "--stage", "stage2",
                        "--out", str(d), "--seed", "0", "--scale", "0.3"], capture_output=True, text=True, cwd=REPO, env=_env(), timeout=120)
    assert r.returncode == 0, r.stderr[-400:]; return d
def test_refuses_memorization(tmp_path):
    d = _data(tmp_path)
    r = subprocess.run([sys.executable, "-m", "tinymind.cli", "train", "--config", "tiny_debug", "--dataset", str(d),
                        "--output", str(tmp_path / "o"), "--stage", "stage2", "--target-tokens", "100000",
                        "--max-source-passes", "0.5", "--overflow", "drop"], capture_output=True, text=True, cwd=REPO, env=_env(), timeout=120)
    assert r.returncode != 0 and "max_source_passes exceeded" in (r.stdout + r.stderr)
def test_default_trains(tmp_path):
    d = _data(tmp_path)
    r = subprocess.run([sys.executable, "-m", "tinymind.cli", "train", "--config", "tiny_debug", "--dataset", str(d),
                        "--output", str(tmp_path / "o"), "--stage", "stage2", "--max-steps", "1", "--warmup-steps", "0",
                        "--no-objective", "--overflow", "drop"], capture_output=True, text=True, cwd=REPO, env=_env(), timeout=180)
    assert r.returncode == 0, (r.stdout + r.stderr)[-600:]
