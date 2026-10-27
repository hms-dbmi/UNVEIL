"""Step 4: score a finished fairtune_pipeline run with the paper's resolution rule.

A task folder written by the pipeline holds, for its latest complete run ``<k>_``:

    <k>_bootstrapTest_concatenate_p_AUC.csv     AUROC and its p-value
    <k>_bootstrapTest_concatenate_metrics.csv   the fairness metrics (first row)
    <k>_bootstrapTest_concatenate_p_biased.csv  each metric's bias-test p-value (row "pval")
    <k>_run_status.json                         the run's status record

The study's resolution rule: the mitigated model shows no significant EOddAbs disparity
and its AUROC is not significantly lower than the baseline's by a paired permutation test.
That rule is implemented on patient-level predictions by ``03_fairness_evaluation/
compare_models.py`` (column ``resolved``). The pipeline's files hold no paired test, so this
scorer applies the fairness half exactly and an AUROC margin as a proxy for the other half:
  * fair:  no listed metric is significant (default ``metrics=('EOddAbs',)``) -- every
           bias-test p-value is at least ``alpha`` (``p_rule='ge'``; 'gt' asks for
           p > alpha), and all of them are present (``require_all_p``);
  * keeps AUROC: AUROC - baseline AUROC >= -``auroc_margin``, the baseline being the
           unmitigated model on the same task (pass its folder or its AUROC).

The p-values are the pipeline's: each tests the run's own group gap against a
shuffled-label null; they do not compare the run with the baseline.

A run whose status record shows a failure is not scored (``status_check``): the pipeline
catches a fold's training error, moves on to the next fold and still writes a complete
bootstrap, so the files alone cannot tell such a run from a finished one.
"""

import glob
import json
import math
import os

import pandas as pd

DEFAULT_METRICS = ("EOddAbs",)
ALL_METRICS = ("EOpp0", "EOpp1", "EOddAbs", "EBAcc", "AUCDiff", "ACC_diff")
REQUIRED = ("bootstrapTest_concatenate_p_AUC.csv", "bootstrapTest_concatenate_metrics.csv")
OPTIONAL = ("bootstrapTest_concatenate_p_biased.csv",)
# keys the pipeline's status writer adds when a fold failed; their PRESENCE condemns a run
FAILURE_KEYS = ("error_class", "error_msg", "traceback", "reason", "baseline_auc")
ERROR_KEYS = ("error_class", "error_msg", "traceback")


def best_run_prefix(task_dir):
    """The prefix of the latest complete run: the largest numeric ``<k>_`` whose two
    required files exist, else the unprefixed legacy set, else None."""
    if not os.path.isdir(task_dir):
        return None
    runs = {}
    for suffix in REQUIRED + OPTIONAL:
        if os.path.exists(os.path.join(task_dir, suffix)):
            runs.setdefault("", set()).add(suffix)
        for f in glob.glob(os.path.join(task_dir, f"*_{suffix}")):
            stem = os.path.basename(f)[: -len(suffix) - 1]
            if stem.isdigit():
                runs.setdefault(stem, set()).add(suffix)
    complete = [p for p, s in runs.items() if p != "" and set(REQUIRED) <= s]
    if complete:
        return f"{max(complete, key=int)}_"
    if "" in runs and set(REQUIRED) <= runs[""]:
        return ""
    return None


def status_problems(task_dir, prefix, ignore_error_keys=False):
    """Why the run's status record says it must not be scored; [] when it is clean.

    A missing record is not a problem (external-cohort inference writes none).

    ``ignore_error_keys`` applies the rule the pipeline's authors use for their own runs:
    error_class / error_msg / traceback are ignored when the status is "success", they are
    the only failure keys, and both bootstrap files were written no earlier than the
    record's ``ended_at``. The rule cannot tell keys left by an earlier attempt that was
    rerun cleanly from a fold that failed in the same attempt and was skipped: both look
    exactly like this. Use it only for runs whose job logs you have checked."""
    path = os.path.join(task_dir, f"{prefix}run_status.json")
    if not os.path.lexists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            st = json.load(fh)
    except (OSError, ValueError) as e:
        return [f"{os.path.basename(path)} cannot be read ({type(e).__name__})"]
    if not isinstance(st, dict):
        return [f"{os.path.basename(path)} is not a JSON object"]
    if st.get("status") != "success":
        return [f"status {st.get('status')!r} at fold {st.get('fold')}"]
    bad = [k for k in FAILURE_KEYS if k in st]
    if bad and ignore_error_keys and all(k in ERROR_KEYS for k in bad):
        ended = st.get("ended_at")
        ok_time = (not isinstance(ended, bool) and isinstance(ended, (int, float))
                   and math.isfinite(ended) and ended > 0)
        try:
            mt = min(os.path.getmtime(os.path.join(task_dir, prefix + s)) for s in REQUIRED)
        except OSError:
            mt = None
        if ok_time and mt is not None and mt >= ended:
            bad = []
    if bad:
        return [f"status 'success' but {', '.join(bad)} present: a fold (or an earlier "
                f"attempt) failed and a later success overwrote the status"]
    return []


def _read_indexed(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, index_col=0, float_precision="round_trip")
    except Exception:                                        # noqa: BLE001
        return None


def _f(v):
    try:
        return float(v) if pd.notna(v) else None
    except (TypeError, ValueError):
        return None


def parse_run(task_dir, metrics=DEFAULT_METRICS, status_check=True, ignore_error_keys=False):
    """The values of one task folder's latest complete run, or None if it has none (or,
    with ``status_check``, if its status record shows a failure)."""
    prefix = best_run_prefix(task_dir)
    if prefix is None:
        return None
    problems = status_problems(task_dir, prefix, ignore_error_keys) if status_check else []
    if problems:
        return {"task_dir": task_dir, "prefix": prefix, "status_problems": problems}
    base = os.path.join(task_dir, f"{prefix}bootstrapTest_concatenate_")
    pauc, met = _read_indexed(base + "p_AUC.csv"), _read_indexed(base + "metrics.csv")
    if pauc is None or met is None or "sensitiveAttr" not in met.columns:
        return None
    rec = {"task_dir": task_dir, "prefix": prefix, "status_problems": [],
           "auroc": None, "auroc_p_value": None}
    col = pauc.columns[0]
    if "AUROC" in pauc.index:
        rec["auroc"] = _f(pauc.loc["AUROC", col])
    if "p_value" in pauc.index:
        rec["auroc_p_value"] = _f(pauc.loc["p_value", col])
    row0 = met.iloc[0]
    pb = _read_indexed(base + "p_biased.csv")
    for m in metrics:
        rec[m] = _f(row0.get(m)) if m in met.columns else None
        rec["p_" + m] = (_f(pb.loc["pval", m]) if pb is not None and "pval" in pb.index
                         and m in pb.columns else None)
    return rec


def resolve(rec, baseline_auroc, metrics=DEFAULT_METRICS, alpha=0.05, p_rule="ge",
            require_all_p=True, auroc_margin=0.02, check_auroc=True):
    """Add 'fair', 'keeps_auroc', 'resolved' (and the reasons) to a parse_run record."""
    if p_rule not in ("ge", "gt"):
        raise ValueError("p_rule must be 'ge' (p >= alpha) or 'gt' (p > alpha)")
    out = dict(rec or {})
    if rec is None or rec.get("status_problems"):
        out.update(fair=False, keeps_auroc=False, resolved=False,
                   why="no complete run" if rec is None else "; ".join(rec["status_problems"]))
        return out
    ps = [rec.get("p_" + m) for m in metrics]
    significant = [m for m, p in zip(metrics, ps)
                   if p is not None and (p < alpha if p_rule == "ge" else p <= alpha)]
    missing = [m for m, p in zip(metrics, ps) if p is None]
    fair = not significant and not (require_all_p and missing)
    if check_auroc:
        if baseline_auroc is None or rec.get("auroc") is None:
            keeps = False
            d = None
        else:
            d = rec["auroc"] - float(baseline_auroc)
            keeps = d >= -auroc_margin
    else:
        keeps, d = True, None
    why = []
    if significant:
        why.append(f"significant: {significant}")
    if missing and require_all_p:
        why.append(f"missing p-values: {missing}")
    if check_auroc and not keeps:
        why.append("no baseline AUROC" if d is None else
                   f"AUROC {d:+.4f} vs baseline (margin -{auroc_margin})")
    out.update(fair=fair, significant=significant, missing_p=missing,
               baseline_auroc=None if baseline_auroc is None else float(baseline_auroc),
               d_auroc=d, keeps_auroc=keeps, resolved=bool(fair and keeps),
               why="; ".join(why))
    return out
