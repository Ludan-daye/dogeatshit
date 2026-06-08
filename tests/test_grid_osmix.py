import csv, pathlib

GRID = pathlib.Path(__file__).resolve().parents[1] / "src/configs/experiment_grid_osmix.csv"


def test_grid_schema_and_values():
    rows = list(csv.DictReader(open(GRID)))
    assert rows, "grid is empty"
    need = {"exp_id", "group", "model", "real_dataset", "ai_dataset",
            "mode", "p_syn", "n_train", "k_max", "seed"}
    assert need.issubset(rows[0].keys()), f"missing columns: {need - set(rows[0].keys())}"
    for r in rows:
        assert r["mode"] in {"constant", "ramp"}, r["mode"]
        assert 0.0 <= float(r["p_syn"]) <= 1.0
        assert int(r["k_max"]) >= 1
    ids = [r["exp_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate exp_id"


def test_has_control_and_ramp():
    rows = list(csv.DictReader(open(GRID)))
    assert any(r["mode"] == "constant" and float(r["p_syn"]) == 0.0 for r in rows), \
        "missing p_syn=0 control (D1)"
    assert any(r["mode"] == "ramp" for r in rows), "missing D2 ramp chain"
