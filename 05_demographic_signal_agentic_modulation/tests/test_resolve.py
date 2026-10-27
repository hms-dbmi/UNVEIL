"""The resolution rule on task folders laid out the way the pipeline writes them."""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import run_module_tests  # noqa: E402

from unveil_tilefilter.resolve import (  # noqa: E402
    ALL_METRICS, DEFAULT_METRICS, best_run_prefix, parse_run, resolve)

PB_COLS = ["PQD", "EOpp0", "EOpp1", "EBAcc", "EOdd", "EOddAbs", "EOddMax", "AUCDiff",
           "ACC_diff"]


def write_run(d, prefix="1_", auroc=0.80, pvals=None, values=None, status=None,
              with_pbiased=True):
    os.makedirs(d, exist_ok=True)
    pvals = dict({c: 0.5 for c in PB_COLS}, **(pvals or {}))
    values = dict({m: 0.1 for m in ALL_METRICS}, **(values or {}))
    with open(os.path.join(d, f"{prefix}bootstrapTest_concatenate_p_AUC.csv"), "w") as fh:
        fh.write(f",0\nAUROC,{auroc!r}\np_value,0.0\n")
    cols = ["sensitiveAttr", "group_type", "N", "AUC"] + list(ALL_METRICS)
    with open(os.path.join(d, f"{prefix}bootstrapTest_concatenate_metrics.csv"), "w") as fh:
        fh.write("," + ",".join(cols) + "\n")
        for i, g in enumerate(("female", "male")):
            fh.write(f"{i},{g},unspecified,100.0,0.7," +
                     ",".join(repr(values[m]) for m in ALL_METRICS) + "\n")
    if with_pbiased:
        with open(os.path.join(d, f"{prefix}bootstrapTest_concatenate_p_biased.csv"),
                  "w") as fh:
            fh.write("," + ",".join(PB_COLS) + "\npval," +
                     ",".join(repr(pvals[c]) for c in PB_COLS) + "\n")
    if status is not None:
        with open(os.path.join(d, f"{prefix}run_status.json"), "w") as fh:
            json.dump(status, fh)


def test_latest_complete_run_is_read():
    with tempfile.TemporaryDirectory() as tmp:
        write_run(tmp, "1_", auroc=0.70)
        write_run(tmp, "2_", auroc=0.75)
        # 3_ is incomplete: only its p_biased file exists
        open(os.path.join(tmp, "3_bootstrapTest_concatenate_p_biased.csv"), "w").close()
        assert best_run_prefix(tmp) == "2_"
        assert parse_run(tmp)["auroc"] == 0.75
        assert best_run_prefix(os.path.join(tmp, "nope")) is None


def test_rule():
    with tempfile.TemporaryDirectory() as tmp:
        write_run(tmp, auroc=0.785, values={"EOddAbs": 0.031})
        rec = parse_run(tmp)
        assert rec["EOddAbs"] == 0.031 and rec["p_EOddAbs"] == 0.5
        r = resolve(rec, 0.80)
        assert r["resolved"] and r["fair"] and r["keeps_auroc"]
        assert abs(r["d_auroc"] + 0.015) < 1e-12
        assert not resolve(rec, 0.806)["resolved"]                   # past the margin
        assert resolve(rec, 0.85, auroc_margin=0.1)["resolved"]
        # d is AUROC - baseline in floating point, compared with -margin, as in the paper's
        # code: 0.78 - 0.80 is -0.020000000000000018, just past -0.02 ...
        write_run(tmp, auroc=0.78)
        assert not resolve(parse_run(tmp), 0.80)["keeps_auroc"]
        # ... while 0.02 - 0.04 is exactly -0.02, which keeps AUROC (>=, not >)
        assert 0.02 - 0.04 == -0.02
        write_run(tmp, auroc=0.02)
        assert resolve(parse_run(tmp), 0.04)["keeps_auroc"]
        assert not resolve(rec, None)["resolved"]                    # no baseline
        assert resolve(rec, None, check_auroc=False)["resolved"]


def test_p_values():
    with tempfile.TemporaryDirectory() as tmp:
        write_run(tmp, pvals={"EOddAbs": 0.05})
        rec = parse_run(tmp)
        assert DEFAULT_METRICS == ("EOddAbs",)
        assert resolve(rec, 0.8)["resolved"]                         # p >= 0.05 is fair
        assert not resolve(rec, 0.8, p_rule="gt")["resolved"]        # p > 0.05 is not met
        write_run(tmp, pvals={"EOddAbs": 0.049})
        r = resolve(parse_run(tmp), 0.8)
        assert not r["resolved"] and r["significant"] == ["EOddAbs"]
        # other metrics count only when asked for
        write_run(tmp, pvals={"AUCDiff": 0.01})
        assert resolve(parse_run(tmp), 0.8)["resolved"]
        assert not resolve(parse_run(tmp, ALL_METRICS), 0.8, ALL_METRICS)["resolved"]


def test_missing_p_values():
    with tempfile.TemporaryDirectory() as tmp:
        write_run(tmp, with_pbiased=False)
        rec = parse_run(tmp)
        assert rec["p_EOddAbs"] is None
        assert not resolve(rec, 0.8)["resolved"]
        assert resolve(rec, 0.8, require_all_p=False)["resolved"]


def test_status_guard():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "a")
        write_run(d, status={"status": "success", "ended_at": 1.0})
        assert resolve(parse_run(d), 0.8)["resolved"]
        write_run(d, status={"status": "running", "fold": 2})
        r = resolve(parse_run(d), 0.8)
        assert not r["resolved"] and "running" in r["why"]
        assert parse_run(d, status_check=False)["auroc"] == 0.8
        # a swallowed fold error: "success" with error keys left in the record
        bad = {"status": "success", "error_class": "RuntimeError", "error_msg": "x",
               "traceback": "...", "ended_at": time.time() + 3600}
        write_run(d, status=bad)
        assert not resolve(parse_run(d), 0.8)["resolved"]
        # the pipeline authors' rule: ignored when both files postdate ended_at
        assert not resolve(parse_run(d, ignore_error_keys=True), 0.8)["resolved"]
        write_run(d, status=dict(bad, ended_at=time.time() - 3600))
        assert resolve(parse_run(d, ignore_error_keys=True), 0.8)["resolved"]
        write_run(d, status=dict(bad, reason="insufficient", ended_at=1.0))
        assert not resolve(parse_run(d, ignore_error_keys=True), 0.8)["resolved"]


if __name__ == "__main__":
    sys.exit(run_module_tests(globals()))
