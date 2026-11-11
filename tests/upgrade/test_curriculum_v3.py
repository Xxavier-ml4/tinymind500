import json
from pathlib import Path
from tinymind.data.curriculum_v2 import load_manifest, effective_mixture
from tinymind.cli_training import load_profile
from tinymind.training.objective import StageObjective
REPO = Path(__file__).resolve().parents[2]
MAN = REPO / "configs" / "curriculum_v3.json"
STAGES = [f"stage{i}" for i in range(1, 8)]
def test_loads():
    m = load_manifest(MAN); assert m["version"] == 3 and set(m["stages"]) == set(STAGES)
def test_mixtures():
    m = load_manifest(MAN)
    for s in STAGES: assert abs(sum(effective_mixture(s, m).values()) - 1) < 1e-9
def test_budget():
    mc, _, _ = load_profile("500m"); m = load_manifest(MAN)
    tot = sum(s["target_tokens"] for s in m["stages"].values())
    assert tot == m["total_target_tokens"] and tot / mc.count_parameters() >= 20
def test_stage1_policy():
    s1 = load_manifest(MAN)["stages"]["stage1"]
    assert s1["corpus"]["byte_share"] > 0.99 and s1["corpus"]["val_records"] >= 20000
    assert "comprehension" not in s1["primary"]
def test_objectives_parse():
    d = REPO / "configs" / "stages_v3"
    for s in STAGES:
        o = StageObjective.from_stage(s, d); assert o.stage == s
        if s >= "stage4":
            assert any(m["name"] == "generation_terminates" for m in json.loads((d / f"{s}.objective.json").read_text())["measurements"])
