"""Compare this package's run parser and resolution rule with the paper's own code.

    python verification/eq_resolve.py --runs-root <repro package>/runs \\
        --old-analysis <repro package>/code/primary/analysis

Every task folder under --runs-root that holds a bootstrap is read twice: by
resolve.parse_run (this package) and by the paper's collect_confirmation.parse_result_dir
(imported from --old-analysis). The two must agree on whether the run can be scored (the
status guard), and on AUROC, its p-value, the six fairness values and their six p-values,
exactly; and the fairness half of the rule (resolve(...)['fair']) must equal the paper's
resolution(rec) strict reading. Exit 1 on any disagreement.

With --manifest (the reproduction package's MANIFEST.csv), every folder that both refuse
is also read at its source_dir with --ignore-error-keys, and the script reports how many
that rule admits there. The source folders keep the original file times the rule
compares; copies elsewhere may not.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
from unveil_tilefilter.resolve import parse_run, resolve  # noqa: E402

# the pre-revision rule of the paper's analysis code, which this script compares against:
# all six of its metrics non-significant (the revised rule is EOddAbs alone, resolve's default)
DEFAULT_METRICS = ("EOpp0", "EOpp1", "EOddAbs", "EOddMax", "EBAcc", "AUCDiff")

OLD = {"auroc": "auroc_overall", "auroc_p_value": "auroc_p_value",
       "EOpp0": "eopp0", "EOpp1": "eopp1", "EOddAbs": "eodd_abs", "EOddMax": "eodd_max",
       "EBAcc": "ebacc", "AUCDiff": "auc_diff",
       "p_EOpp0": "p_eopp0", "p_EOpp1": "p_eopp1", "p_EOddAbs": "p_eodd_abs",
       "p_EOddMax": "p_eodd_max", "p_EBAcc": "p_ebacc", "p_AUCDiff": "p_auc_diff"}


def task_dirs(root):
    for dirpath, _dirs, files in os.walk(root):
        if any(f.endswith("bootstrapTest_concatenate_p_AUC.csv") for f in files):
            yield dirpath


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--old-analysis", required=True)
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.old_analysis))
    import collect_confirmation as CC

    n = scored = refused = disagree = 0
    refused_dirs = []
    for d in sorted(task_dirs(a.runs_root)):
        n += 1
        old = CC.parse_result_dir(d)
        new = parse_run(d, DEFAULT_METRICS)
        new_ok = new is not None and not new.get("status_problems")
        problems = []
        if (old is not None) != new_ok:
            problems.append(f"scored: old {old is not None}, new {new_ok}")
        elif old is not None:
            scored += 1
            for k_new, k_old in OLD.items():
                if new.get(k_new) != old.get(k_old):
                    problems.append(f"{k_new}: old {old.get(k_old)!r} new {new.get(k_new)!r}")
            _lit, strict, _miss = CC.resolution(old)
            fair = resolve(new, None, metrics=DEFAULT_METRICS, check_auroc=False)["fair"]
            if fair != strict:
                problems.append(f"fair: old {strict}, new {fair}")
        else:
            refused += 1
            refused_dirs.append(d)
        if problems:
            disagree += 1
            print(f"DIFF {os.path.relpath(d, a.runs_root)}: {problems[:4]}")
    print(f"\n{n} task folders: {scored} scored by both, {refused} refused by both "
          f"(status), {disagree} disagreements")
    if a.manifest and refused_dirs:
        root = os.path.dirname(os.path.abspath(a.manifest))
        with open(a.manifest, newline="") as fh:
            src = {r["run_dir"].rstrip("/"): r["source_dir"] for r in csv.DictReader(fh)}
        admitted = 0
        for d in refused_dirs:
            sd = src.get(os.path.relpath(d, root))
            r = parse_run(sd, DEFAULT_METRICS, ignore_error_keys=True) if sd else None
            ok = r is not None and not r.get("status_problems")
            admitted += ok
            if not ok:
                print(f"not admitted at source: {os.path.relpath(d, root)} ({sd})")
        print(f"--ignore-error-keys on their source folders: {admitted} of {len(refused_dirs)} "
              f"refused folders admitted")
    return 1 if disagree or not n else 0


if __name__ == "__main__":
    sys.exit(main())
