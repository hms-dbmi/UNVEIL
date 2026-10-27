import numpy as np
import pytest

from unveil.fairness import GROUP_METRICS, METRICS, OVERALL_METRICS, auc, bias_test, bootstrap_metrics, fairness_metrics, paired_swap_test


def test_auc_matches_pairwise_definition_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    score = np.round(rng.normal(size=300) + y, 1)
    difference = score[y == 1][:, None] - score[y == 0][None, :]
    assert auc(y, score) == pytest.approx(np.mean((difference > 0) + 0.5 * (difference == 0)))


def test_disparities_are_best_minus_worst_group_per_metric():
    y = np.array([1, 1, 0, 0] * 3)
    pred = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1])
    group = np.repeat(["a", "b", "c"], 4)
    m = fairness_metrics(y, pred.astype(float), pred, group)
    assert m["EOpp1"] == pytest.approx(1.0 - 0.5)
    assert m["EOpp0"] == pytest.approx(1.0 - 0.0)
    assert m["EBAcc"] == pytest.approx(1.0 - 0.5)
    assert m["EOddAbs"] == pytest.approx((0.5 + 1.0) / 2)
    assert m["ACCDiff"] == pytest.approx(1.0 - 0.5)
    assert m["AUCDiff"] == pytest.approx(1.0 - 0.5)
    privileged = fairness_metrics(y, pred.astype(float), pred, group, privileged="b")
    assert privileged["EOpp1"] == pytest.approx(0.5 - 1.0)
    assert privileged["EOpp0"] == pytest.approx(1.0 - 0.0)


@pytest.mark.parametrize("null", ["bootstrap", "permutation"])
def test_bias_test_controls_false_positives_and_detects_a_gap(null):
    rng = np.random.default_rng(1)
    p_null, p_gap = [], []
    for _ in range(100):
        n = 160
        y = rng.integers(0, 2, n)
        group = rng.integers(0, 2, n)
        score = rng.normal(size=n) + y
        p_null.append(bias_test(y, score, (score > 0.5).astype(int), group, np.arange(n), 99, rng, null=null)["EOpp1"])
        shifted = score - 1.5 * group * y
        p_gap.append(bias_test(y, shifted, (shifted > 0.5).astype(int), group, np.arange(n), 99, rng, null=null)["EOpp1"])
    assert np.mean(np.array(p_null) < 0.05) <= 0.1
    assert np.mean(np.array(p_gap) < 0.05) >= 0.8


def test_patient_with_two_units_stays_in_one_group():
    rng = np.random.default_rng(2)
    patient = np.repeat(np.arange(60), 2)
    y = np.tile([0, 1], 60)
    group = np.repeat(rng.integers(0, 2, 60), 2)
    score = rng.normal(size=120) + y
    pred = (score > 0.5).astype(int)
    metric_draws, group_draws = bootstrap_metrics(y, group, patient, [(score, pred)], 50, rng)
    assert metric_draws.shape == (50, 1, len(METRICS) + len(OVERALL_METRICS))
    assert group_draws.shape == (50, 1, 2, len(GROUP_METRICS))
    assert np.isfinite(metric_draws).all()
    p = bias_test(y, score, pred, group, patient, 50, rng)
    assert set(p) == set(METRICS)
    assert all(0 < value <= 1 for value in p.values())


def test_paired_swap_test_on_identical_models_finds_no_change():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 80)
    group = rng.integers(0, 2, 80)
    score = rng.normal(size=80) + y
    pred = (score > 0.5).astype(int)
    change, p, p_lower = paired_swap_test(y, group, np.arange(80), [(score, pred), (score, pred)], 50, rng)
    assert np.allclose(change, 0)
    assert np.allclose(p, 1)
    assert np.allclose(p_lower, 1)
