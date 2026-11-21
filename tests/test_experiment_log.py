from hitflop.experiment_log import log_experiment, read_log


def test_log_roundtrip(tmp_path):
    import numpy as np
    p = tmp_path / "log.jsonl"
    log_experiment("x", "phase0", {"seed": 42}, {"auc": np.float64(0.7), "arr": np.arange(3)}, "note", path=p)
    log_experiment("y", "phase0", {}, {}, path=p)
    recs = read_log(p)
    assert len(recs) == 2 and recs[0]["results"]["auc"] == 0.7 and recs[0]["results"]["arr"] == [0, 1, 2]
