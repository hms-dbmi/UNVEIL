import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata

METRICS = ("EOpp0", "EOpp1", "EBAcc", "EOddAbs", "ACCDiff", "AUCDiff")
DERIVED_METRICS = ("EOddAbs",)
GROUP_METRICS = ("AUC", "TPR", "TNR", "BAcc", "ACC")
OVERALL_METRICS = ("AUC_overall", "BAcc_overall")
PATIENT_KEYS = ["cohort", "cancer", "task", "patient_id"]


def patient_predictions(predictions, threshold):
    units = predictions.groupby(PATIENT_KEYS + ["label"], sort=True)["probability"].mean().reset_index()
    units["label"] = units["label"].astype(int)
    units["prediction"] = (units["probability"] >= threshold).astype(int)
    return units


def _patient_counts(frame, labels):
    return frame.groupby(labels)["patient_id"].nunique()


def fairness_groups(frame, exposure, min_group_size, pairwise=False):
    """Yield (contrast, rows, group labels, privileged level) for one exposure."""
    if exposure.kind == "continuous":
        a = pd.to_numeric(frame[exposure.name], errors="coerce")
        keep = a.notna().to_numpy()
        median = a[keep].groupby(frame.loc[keep, "patient_id"]).first().median()
        labels = np.where(a[keep] > median, "above_median", "below_median")
        if len(np.unique(labels)) == 2 and _patient_counts(frame.loc[keep], labels).min() >= min_group_size:
            yield "above_vs_below_median", frame.loc[keep], labels, "below_median"
        return
    values = frame[exposure.name].astype("string")
    counts = _patient_counts(frame[values.notna().to_numpy()], values.dropna().to_numpy())
    levels = [v for v in (exposure.reference,) + tuple(exposure.levels) if counts.get(v, 0) >= min_group_size]
    if len(levels) < 2:
        return
    keep = values.isin(levels).fillna(False).to_numpy()
    privileged = exposure.reference if exposure.reference in levels else None
    yield "all_groups", frame.loc[keep], values[keep].to_numpy(dtype=str), privileged
    if pairwise and privileged is not None and len(levels) > 2:
        for level in levels[1:]:
            pair = values.isin([privileged, level]).fillna(False).to_numpy()
            yield f"{level}_vs_{privileged}", frame.loc[pair], values[pair].to_numpy(dtype=str), privileged


def auc(y, score):
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(score)
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _rates(y, score, pred):
    positive = y == 1
    tpr = pred[positive].mean() if positive.any() else np.nan
    tnr = 1 - pred[~positive].mean() if (~positive).any() else np.nan
    return auc(y, score), tpr, tnr, (tpr + tnr) / 2, (pred == y).mean()


def _group_matrix(y, score, pred, group, levels):
    return np.array([_rates(y[group == g], score[group == g], pred[group == g]) for g in levels])


def _disparities(matrix, levels, privileged):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if privileged is None:
            best, worst = np.nanmax(matrix, axis=0), np.nanmin(matrix, axis=0)
        else:
            i = list(levels).index(privileged)
            best, worst = matrix[i], np.nanmin(np.delete(matrix, i, axis=0), axis=0)
    d_auc, d_tpr, d_tnr, d_bacc, d_acc = best - worst
    return np.array([d_tnr, d_tpr, d_bacc, (abs(d_tnr) + abs(d_tpr)) / 2, d_acc, d_auc])


def _metric_vector(y, score, pred, group, levels, privileged):
    matrix = _group_matrix(y, score, pred, group, levels)
    overall = _rates(y, score, pred)
    return np.r_[_disparities(matrix, levels, privileged), overall[0], overall[3]], matrix


def group_table(y, score, pred, group):
    levels = np.unique(group)
    table = pd.DataFrame(_group_matrix(y, score, pred, group, levels), index=levels, columns=GROUP_METRICS)
    table.insert(0, "n_positive", [int(y[group == g].sum()) for g in levels])
    table.insert(0, "n_units", [int((group == g).sum()) for g in levels])
    return table


def fairness_metrics(y, score, pred, group, privileged=None):
    """Best minus worst group (or privileged minus worst other group) for each metric."""
    levels = np.unique(group)
    vector, _ = _metric_vector(y, score, pred, group, levels, privileged)
    return dict(zip(METRICS + OVERALL_METRICS, vector))


def _clusters(cluster):
    _, first, inverse = np.unique(cluster, return_index=True, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    rows_of = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    return first, rows_of


def _within(blocks, rng, replace):
    drawn = np.empty(sum(len(b) for b in blocks), dtype=int)
    for block in blocks:
        drawn[block] = rng.choice(block, len(block), replace=replace)
    return drawn


def bias_test(y, score, pred, group, cluster, n_boot, rng, privileged=None, null="permutation", strata=None):
    """One-sided p-value that each disparity is larger than expected with no group effect.

    Patient slots keep their group label while the outcomes filling them are redrawn,
    without replacement ("permutation", the study null, within `strata` such as the outcome
    class) or with replacement ("bootstrap").
    """
    levels = np.unique(group)
    observed, _ = _metric_vector(y, score, pred, group, levels, privileged)
    observed = observed[:len(METRICS)]
    first, rows_of = _clusters(cluster)
    slot_group = group[first]
    if strata is None:
        blocks = [np.arange(len(first))]
    else:
        stratum = pd.Series(strata).astype(str).to_numpy()[first]
        blocks = [np.flatnonzero(stratum == s) for s in np.unique(stratum)]
    draws = np.empty((n_boot, len(METRICS)))
    for b in range(n_boot):
        drawn = _within(blocks, rng, replace=null == "bootstrap")
        rows = np.concatenate([rows_of[j] for j in drawn])
        labels = np.repeat(slot_group, [len(rows_of[j]) for j in drawn])
        draws[b] = _disparities(_group_matrix(y[rows], score[rows], pred[rows], labels, levels), levels, privileged)
    valid = np.isfinite(draws)
    exceed = (draws >= observed - 1e-12) & valid
    p = (exceed.sum(axis=0) + 1) / (valid.sum(axis=0) + 1)
    return dict(zip(METRICS, np.where(np.isfinite(observed), p, np.nan)))


def bootstrap_metrics(y, group, cluster, models, n_boot, rng, privileged=None):
    """Resample patients within each group; returns metric and per-group draws for each model."""
    levels = np.unique(group)
    first, rows_of = _clusters(cluster)
    blocks = [np.flatnonzero(group[first] == g) for g in levels]
    metric_draws = np.empty((n_boot, len(models), len(METRICS) + len(OVERALL_METRICS)))
    group_draws = np.empty((n_boot, len(models), len(levels), len(GROUP_METRICS)))
    for b in range(n_boot):
        rows = np.concatenate([rows_of[j] for j in _within(blocks, rng, replace=True)])
        for i, (score, pred) in enumerate(models):
            metric_draws[b, i], group_draws[b, i] = _metric_vector(y[rows], score[rows], pred[rows], group[rows], levels, privileged)
    return metric_draws, group_draws


def auroc_test(y, score, n_boot, rng):
    """One-sided p-value that the AUROC exceeds that of labels and scores drawn independently."""
    observed = auc(y, score)
    n = len(y)
    null = np.array([auc(y[rng.integers(n, size=n)], score[rng.integers(n, size=n)]) for _ in range(n_boot)])
    valid = np.isfinite(null)
    return observed, (np.sum(null[valid] >= observed) + 1) / (valid.sum() + 1)


def paired_swap_test(y, group, cluster, models, n_perm, rng, privileged=None):
    """Change between two models with its two-sided p-value and the one-sided (lower-tail)
    p-value that the change is not below zero, swapping the models' predictions within patients."""
    levels = np.unique(group)
    (score_a, pred_a), (score_b, pred_b) = models
    observed = _metric_vector(y, score_b, pred_b, group, levels, privileged)[0] - _metric_vector(y, score_a, pred_a, group, levels, privileged)[0]
    _, inverse = np.unique(cluster, return_inverse=True)
    null = np.empty((n_perm, observed.size))
    for k in range(n_perm):
        swap = rng.random(inverse.max() + 1)[inverse] < 0.5
        first = _metric_vector(y, np.where(swap, score_b, score_a), np.where(swap, pred_b, pred_a), group, levels, privileged)[0]
        second = _metric_vector(y, np.where(swap, score_a, score_b), np.where(swap, pred_a, pred_b), group, levels, privileged)[0]
        null[k] = second - first
    valid = np.isfinite(null)
    exceed = (np.abs(null) >= np.abs(observed) - 1e-12) & valid
    lower = (null <= observed + 1e-12) & valid
    p = (exceed.sum(axis=0) + 1) / (valid.sum(axis=0) + 1)
    p_lower = (lower.sum(axis=0) + 1) / (valid.sum(axis=0) + 1)
    finite = np.isfinite(observed)
    return observed, np.where(finite, p, np.nan), np.where(finite, p_lower, np.nan)
