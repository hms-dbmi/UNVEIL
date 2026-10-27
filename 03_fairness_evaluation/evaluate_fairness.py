import argparse

import numpy as np
import pandas as pd

from unveil.cli import DEFAULT_SPEC
from unveil.config import load_spec
from unveil.data import attach_clinical, read_table, write_table
from unveil.fairness import (DERIVED_METRICS, GROUP_METRICS, METRICS, OVERALL_METRICS, PATIENT_KEYS, auroc_test, bias_test,
                             bootstrap_metrics, fairness_groups, fairness_metrics, group_table, patient_predictions)
from unveil.multiplicity import add_fdr

FDR_FAMILY = ("cohort", "exposure", "contrast", "metric")


def interval(draws):
    return np.nanquantile(draws, [0.025, 0.975], axis=0) if np.isfinite(draws).any() else np.full((2,) + draws.shape[1:], np.nan)


def evaluate_cell(frame, group, privileged, args, rng):
    y = frame["label"].to_numpy()
    score = frame["probability"].to_numpy()
    pred = frame["prediction"].to_numpy()
    patient = frame["patient_id"].to_numpy()
    strata = frame[args.stratify_by].astype(str).agg("|".join, axis=1).to_numpy() if args.stratify_by else None
    observed = fairness_metrics(y, score, pred, group, privileged)
    p_values = bias_test(y, score, pred, group, patient, args.n_boot, rng, privileged, args.null, strata)
    metric_draws, group_draws = bootstrap_metrics(y, group, patient, [(score, pred)], args.n_boot, rng, privileged)
    metric_ci, group_ci = interval(metric_draws[:, 0]), interval(group_draws[:, 0])

    disparities = [
        {"metric": m, "derived": m in DERIVED_METRICS, "value": observed[m], "ci_low": metric_ci[0, j], "ci_high": metric_ci[1, j], "p_value": p_values[m]}
        for j, m in enumerate(METRICS)
    ]
    table = group_table(y, score, pred, group)
    groups = []
    for i, (level, row) in enumerate(table.iterrows()):
        entry = {"group": level, "n_units": row["n_units"], "n_positive": row["n_positive"]}
        for j, m in enumerate(GROUP_METRICS):
            entry.update({m: row[m], f"{m}_ci_low": group_ci[0, i, j], f"{m}_ci_high": group_ci[1, i, j]})
        groups.append(entry)
    overall_auc, auc_p = auroc_test(y, score, args.n_boot, rng)
    offset = len(METRICS)
    groups.append({
        "group": "overall", "n_units": len(y), "n_positive": int(y.sum()),
        "AUC": overall_auc, "AUC_ci_low": metric_ci[0, offset], "AUC_ci_high": metric_ci[1, offset], "AUC_p_value": auc_p,
        "BAcc": observed[OVERALL_METRICS[1]], "BAcc_ci_low": metric_ci[0, offset + 1], "BAcc_ci_high": metric_ci[1, offset + 1],
    })
    return disparities, groups


def main():
    parser = argparse.ArgumentParser(description="Six fairness disparities across demographic groups with bootstrap bias tests and intervals.")
    parser.add_argument("--predictions", required=True, help="slide or patient predictions: cohort, cancer, task, patient_id, probability, label")
    parser.add_argument("--clinical", required=True, help="patient table with exposures")
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--exposures", nargs="+", default=None, help="defaults to the demographic covariate block")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--privileged-reference", action="store_true", help="contrast the exposure reference level against the worst other group instead of best against worst")
    parser.add_argument("--pairwise", action="store_true", help="also test each level against the reference for exposures with more than two groups")
    parser.add_argument("--null", choices=["bootstrap", "permutation"], default="permutation", help="redraw outcomes into group slots without (the study default) or with replacement")
    parser.add_argument("--stratify-by", nargs="+", default=["label"], help="columns within whose levels outcomes are redrawn under the null; the study permutes within outcome classes")
    parser.add_argument("--min-group-size", type=int, default=None, help="minimum patients per group; defaults to the spec")
    parser.add_argument("--n-boot", type=int, default=2000, help="null draws and bootstrap replicates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fdr-family", nargs="+", default=list(FDR_FAMILY))
    parser.add_argument("--output", required=True, help="one row per task, exposure, contrast and disparity metric")
    parser.add_argument("--group-output", default=None, help="per-group and overall performance with intervals")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    min_group_size = spec.min_group_size if args.min_group_size is None else args.min_group_size

    patients = attach_clinical(patient_predictions(read_table(args.predictions), args.threshold), read_table(args.clinical), spec)
    rng = np.random.default_rng(args.seed)
    results, groups = [], []
    for key, cell in patients.groupby(PATIENT_KEYS[:3], sort=True):
        for exposure in map(spec.exposure, args.exposures or spec.covariates["demographic"]):
            for contrast, frame, group, reference in fairness_groups(cell, exposure, min_group_size, args.pairwise):
                privileged = reference if args.privileged_reference else None
                label = {**dict(zip(PATIENT_KEYS[:3], key)), "exposure": exposure.name, "contrast": contrast,
                         "groups": ";".join(np.unique(group)), "privileged": privileged or "best_group"}
                rows, summary = evaluate_cell(frame, group, privileged, args, rng)
                results.extend({**label, **row} for row in rows)
                groups.extend({**label, **row} for row in summary)
    if not results:
        raise SystemExit("no task had at least two groups of the minimum size")
    write_table(add_fdr(pd.DataFrame(results), args.fdr_family), args.output)
    if args.group_output:
        write_table(pd.DataFrame(groups), args.group_output)


if __name__ == "__main__":
    main()
