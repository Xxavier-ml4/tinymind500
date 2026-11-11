from pathlib import Path
from tinymind.cli_training import load_profile, _profile_objective_dir
def test_exact_parameter_count():
    mc, _, _ = load_profile("500m"); assert mc.count_parameters() == 501_090_560
def test_geometry():
    mc, _, _ = load_profile("500m")
    assert (mc.hidden_size, mc.num_layers, mc.num_heads, mc.num_kv_heads) == (1280, 26, 20, 4)
    assert mc.hidden_size % mc.num_heads == 0 and mc.num_heads % mc.num_kv_heads == 0
    assert mc.intermediate_size == 3584 and mc.vocab_size == 32000 and mc.max_seq_len == 2048 and mc.tie_embeddings
def test_policy():
    mc, tok, tr = load_profile("500m")
    assert mc.dropout == 0.0 and tok["type"] == "bpe"
    assert tr["batch_size"] * tr["gradient_accumulation_steps"] * tr["max_seq_len"] == 524_288
def test_objective_dir():
    d = _profile_objective_dir("500m")
    assert d and Path(d).name == "stages_v3" and (Path(d) / "stage1.objective.json").is_file()
