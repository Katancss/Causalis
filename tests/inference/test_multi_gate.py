from copy import deepcopy

import matplotlib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import LinearRegression, LogisticRegression

matplotlib.use("Agg")

from causalis.data_contracts import CausalData, MultiCausalData, MultiGateEstimate
from causalis.scenarios.gate import gate_plot
from causalis.scenarios.multi_unconfoundedness import MultiTreatmentIRM
from causalis.scenarios.unconfoundedness.model import IRM


def _fit(n_arms=3, store_diagnostics=True, user_id=True):
    rng = np.random.default_rng(817)
    n = 240
    x = rng.normal(size=n)
    labels = np.tile(np.arange(n_arms), int(np.ceil(n / n_arms)))[:n]
    rng.shuffle(labels)
    d = np.eye(n_arms, dtype=int)[labels]
    y = 0.7 * x + labels * (1 + 0.4 * x) + rng.normal(size=n)
    df = pd.DataFrame({"id": [f"u{i}" for i in range(n)], "x": x, "y": y})
    arms = [f"d{k}" for k in range(n_arms)]
    df[arms] = d
    data = MultiCausalData(
        df=df,
        outcome="y",
        treatment_names=arms,
        control_treatment="d0",
        confounders=["x"],
        user_id="id" if user_id else None,
    )
    model = MultiTreatmentIRM(
        data,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=500),
        n_folds=3,
        random_state=27,
        store_diagnostics=store_diagnostics,
    ).fit()
    groups = pd.Series(
        np.where(x >= 0, "high", "low"),
        index=pd.Index(df.id, name="id"),
        name="segment",
    )
    return model, groups


@pytest.fixture(scope="module")
def fitted():
    return _fit()


def _manual(model, groups, score, cov_type):
    """Direct scalar moments and a dense joint sandwich, independent of implementation."""
    labels = groups.to_numpy()
    y, d = model._y, model._d
    mu, e = model.g_hat_, model.m_hat_
    group_names = sorted(set(labels))
    n_active = d.shape[1] - 1
    values = np.zeros((len(group_names), n_active))
    a = np.zeros((len(y), len(group_names) * n_active))
    for g, group in enumerate(group_names):
        rows = np.flatnonzero(labels == group)
        for k in range(1, d.shape[1]):
            numerator = []
            for i in rows:
                if score == "GATE":
                    value = mu[i, k] - mu[i, 0]
                    value += d[i, k] * (y[i] - mu[i, k]) / e[i, k]
                    value -= d[i, 0] * (y[i] - mu[i, 0]) / e[i, 0]
                else:
                    value = d[i, k] * (y[i] - mu[i, 0])
                    value -= d[i, 0] * e[i, k] / e[i, 0] * (y[i] - mu[i, 0])
                numerator.append(value)
            denom = len(rows) if score == "GATE" else d[rows, k].sum()
            theta = sum(numerator) / denom
            values[g, k - 1] = theta
            for i, num in zip(rows, numerator):
                design = 1 if score == "GATE" else d[i, k]
                contribution = (num - design * theta) / denom
                if cov_type == "HC1":
                    contribution *= np.sqrt(len(y) / (len(y) - len(group_names)))
                elif cov_type in {"HC2", "HC3"}:
                    exponent = 0.5 if cov_type == "HC2" else 1
                    contribution /= (1 - design / denom) ** exponent
                a[i, g * n_active + k - 1] = contribution
    return values, a.T @ a


@pytest.mark.parametrize("score", ["GATE", "GATET"])
@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3"])
def test_joint_moments_and_covariance(fitted, score, cov_type):
    model, groups = fitted
    result = model.estimate(score, groups=groups, cov_type=cov_type)
    expected, covariance = _manual(model, groups, score, cov_type)
    assert isinstance(result, MultiGateEstimate)
    assert_allclose(result.values, expected)
    assert_allclose(result.covariance, covariance, atol=1e-14)
    assert_allclose(result.std_errors.ravel() ** 2, np.diag(covariance))
    assert_allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-14
    assert abs(covariance[0, 1]) > 1e-5
    assert_array_equal(covariance[:2, 2:], 0)
    assert result.values.shape == (2, 2)
    assert result.covariance.index.names == ["group", "treatment"]
    assert result.summary().shape[0] == 4
    assert_array_equal(
        result.n_treated + result.n_other + result.n_control[:, None],
        np.broadcast_to(result.n_group[:, None], (2, 2)),
    )
    assert_allclose(result.summary()["mean_phi"], result.values.ravel())


@pytest.mark.parametrize("score,global_score", [("GATE", "ATE"), ("GATET", "ATTE")])
def test_one_group_and_no_model_mutation(fitted, monkeypatch, score, global_score):
    model, groups = deepcopy(fitted)
    global_result = model.estimate(global_score)
    snapshot = {
        key: deepcopy(getattr(model, key))
        for key in (
            "score",
            "coef_",
            "se_",
            "psi_",
            "psi_b_",
            "psi_a_",
            "summary_",
            "normalize_ipw_effective_",
            "g_hat_",
            "m_hat_",
        )
    }
    monkeypatch.setattr(model, "fit", lambda *a, **kw: pytest.fail("Unexpected refit"))
    constant_groups = groups.copy()
    constant_groups[:] = "all"
    subgroup_result = model.estimate(score, groups=constant_groups)
    assert_allclose(subgroup_result.values[0], global_result.value)
    getattr(model, score.lower())(groups)
    for key, value in snapshot.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(getattr(model, key), value)
        else:
            assert_array_equal(getattr(model, key), value)


@pytest.mark.parametrize("score", ["GATE", "GATET"])
@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3"])
def test_binary_reduction(score, cov_type):
    multi, groups = _fit(n_arms=2)
    df = multi.data.df.copy()
    binary = IRM(
        CausalData(df=df, treatment="d1", outcome="y", confounders=["x"], user_id="id"),
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(max_iter=500),
        n_folds=3,
        random_state=27,
        overlap_threshold=1e-3,
    ).fit()
    binary.g0_hat_ = multi.g_hat_[:, 0].copy()
    binary.g1_hat_ = multi.g_hat_[:, 1].copy()
    binary.m_hat_ = multi.m_hat_[:, 1].copy()
    expected = binary.estimate(score=score, groups=groups, cov_type=cov_type)
    result = multi.estimate(
        score=score, groups=groups, cov_type=cov_type
    ).for_treatment("d1")
    for field in (
        "values",
        "std_errors",
        "p_values",
        "ci_lower",
        "ci_upper",
        "covariance",
        "n_treated",
        "n_control",
        "mean_phi",
        "std_phi",
    ):
        assert_allclose(getattr(result, field), getattr(expected, field), atol=1e-12)


def test_extraction_contrasts_and_plot(fitted):
    model, groups = fitted
    result = model.gate(groups)
    arm = result.for_treatment("d1")
    contrast = arm.contrast(*result.group_names)
    expected_variance = result.covariance.iloc[0, 0] + result.covariance.iloc[2, 2]
    assert_allclose(contrast.std_error**2, expected_variance)
    assert arm.pairwise_summary(p_adjust="holm").shape[0] == 1
    assert arm.diagnostic_data["orthogonal_signal"].ndim == 1
    assert arm.summary().shape[0] == 2
    figure = gate_plot(arm)
    import matplotlib.pyplot as plt

    plt.close(figure)
    arm.values[0] = -999
    arm.model_options["group_warning_messages"].append("changed")
    assert result.values[0, 0] != -999
    assert "changed" not in result.model_options["group_warning_messages"]
    with pytest.raises(ValueError, match="Unknown active treatment"):
        result.for_treatment("d0")


@pytest.mark.parametrize("score", ["GATE", "GATET"])
def test_group_representations_and_id_alignment(fitted, score):
    model, groups = fitted
    expected = model.estimate(score, groups=groups)
    row_groups = groups.copy()
    row_groups.index = model._fit_row_index_
    variants = [
        groups.sample(frac=1, random_state=3),
        groups.to_frame(),
        row_groups,
        groups.to_numpy(),
        groups.astype("category"),
        pd.get_dummies(groups),
    ]
    for variant in variants:
        result = model.estimate(score, groups=variant)
        assert_allclose(result.values, expected.values)
        assert_allclose(result.covariance, expected.covariance)
    superset = pd.concat(
        [groups, pd.Series(["high"], index=["extra"], name=groups.name)]
    )
    assert_allclose(model.estimate(score, groups=superset).values, expected.values)


def test_fit_identity_survives_mutation(fitted):
    model, groups = deepcopy(fitted)
    expected = model.gate(groups)
    row_groups = groups.copy()
    row_groups.index = model.data.df.index
    model.data.df = model.data.df.iloc[::-1].reset_index(drop=True)
    model.data.treatment_names = ["d0", "d2", "d1"]
    result = model.gate(groups)
    assert result.treatment_names == ["d1", "d2"]
    assert_allclose(result.values, expected.values)
    with pytest.raises(ValueError, match="mapping|fit-time"):
        model.gate(row_groups)
    with pytest.raises(ValueError, match="mapping|fit-time"):
        model.gate(groups.to_numpy())


@pytest.mark.parametrize("store", [False, True])
@pytest.mark.parametrize("requested", [False, True])
def test_diagnostic_gating(store, requested):
    model, groups = _fit(store_diagnostics=store)
    model.store_diagnostics = not store  # Fit-time policy remains authoritative.
    result = model.gatet(groups, diagnostic_data=requested)
    assert (result.diagnostic_data is not None) == (store and requested)
    assert (result.for_treatment("d1").diagnostic_data is not None) == (
        store and requested
    )


@pytest.mark.parametrize("score", ["GATE", "GATET"])
def test_normalization_ignored(fitted, score):
    model, groups = deepcopy(fitted)
    expected = model.estimate(score, groups=groups)
    model.normalize_ipw = True
    with pytest.warns(RuntimeWarning, match="normalize_ipw=True is ignored"):
        result = model.estimate(score, groups=groups)
    assert_allclose(result.values, expected.values)
    assert_allclose(result.covariance, expected.covariance)
    assert result.model_options["normalize_ipw_requested"]
    assert not result.model_options["normalize_ipw_effective"]


def _small_group(model, groups, counts):
    groups = groups.copy()
    groups[:] = "rest"
    for arm, count in enumerate(counts):
        rows = np.flatnonzero(model._d[:, arm])[:count]
        groups.iloc[rows] = "small"
    return groups


def test_sparse_support(fitted):
    model, groups = fitted
    no_active = _small_group(model, groups, [4, 3, 0])
    for score in ("GATE", "GATET"):
        with pytest.raises(ValueError, match="segment=small.*d2"):
            model.estimate(score, groups=no_active)
    no_baseline = _small_group(model, groups, [0, 3, 3])
    with pytest.raises(ValueError, match="baseline"):
        model.gate(no_baseline)
    with pytest.warns(RuntimeWarning, match="no baseline"):
        result = model.gatet(no_baseline)
    assert_allclose(result.values, _manual(model, no_baseline, "GATET", "HC3")[0])
    assert_array_equal(result.n_control, [80, 0])
    assert result.summary().query("group == 'segment=small'")["n_other"].min() == 3


@pytest.mark.parametrize("cov_type", ["HC2", "HC3"])
def test_undefined_inference_only_affects_sparse_cell(fitted, cov_type):
    model, groups = fitted
    sparse = _small_group(model, groups, [3, 1, 3])
    with pytest.warns(RuntimeWarning, match="leverage 1"):
        result = model.gatet(sparse, cov_type=cov_type)
    assert np.isfinite(result.values).all()
    assert np.isnan(result.std_errors[1, 0])
    assert np.isfinite(result.std_errors[1, 1])
    assert np.isnan(result.covariance.iloc[2]).all()
    assert np.isnan(result.covariance.iloc[:, 2]).all()
    valid = [0, 1, 3]
    assert np.isfinite(result.covariance.iloc[valid, valid]).all().all()
    assert not result.summary().loc[2, "is_significant"]


@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3"])
def test_singleton_group(cov_type):
    model, groups = _fit(n_arms=2)
    singleton = _small_group(model, groups, [0, 1])
    with pytest.warns(RuntimeWarning):
        result = model.gatet(singleton, cov_type=cov_type)
    assert np.isfinite(result.values).all()
    assert np.isnan(result.std_errors[1, 0])


def test_validation(fitted):
    model, groups = fitted
    with pytest.raises(ValueError, match="explicit groups"):
        model.estimate("GATE")
    with pytest.raises(ValueError, match="cov_type"):
        model.gate(groups, cov_type="cluster")
    with pytest.raises(ValueError, match="alpha"):
        model.gate(groups, alpha=0)
    without_ids, groups_without_ids = _fit(user_id=False)
    with pytest.raises(ValueError, match="user_id at fit time"):
        without_ids.gate(groups_without_ids)
    for bad in (groups.iloc[:-1], groups.set_axis(["duplicate"] * len(groups))):
        with pytest.raises(ValueError, match="align|unique"):
            model.gate(bad)
    missing = groups.copy()
    missing.iloc[0] = None
    with pytest.raises(ValueError, match="missing"):
        model.gate(missing)
    overlapping = pd.DataFrame(
        {"a": np.ones(len(groups)), "b": np.ones(len(groups))}, index=groups.index
    )
    with pytest.raises(ValueError, match="exclusive"):
        model.gate(overlapping)
    empty = groups.astype("category").cat.add_categories("empty")
    with pytest.raises(ValueError, match="without observations"):
        model.gate(empty)


def test_result_shape_validation(fitted):
    model, groups = fitted
    result = model.gate(groups)
    payload = {name: getattr(result, name) for name in type(result).model_fields}
    with pytest.raises(ValueError, match="values must have shape"):
        MultiGateEstimate(**{**payload, "values": result.values.ravel()})
    with pytest.raises(ValueError, match="group-major"):
        MultiGateEstimate(**{**payload, "covariance": result.covariance.iloc[::-1]})


def test_propensity_denominators_and_stability(fitted):
    model, groups = deepcopy(fitted)
    model.m_hat_[:, 1] = 0
    # GATET uses the active propensity only in the numerator.
    result = model.gatet(groups)
    assert np.isfinite(result.values).all()
    with pytest.raises(RuntimeError, match="denominators"):
        model.gate(groups)
    model.m_hat_[:, 0] = 0
    with pytest.raises(RuntimeError, match="denominators"):
        model.gatet(groups)
    model.m_hat_[:, 0] = np.nan
    with pytest.raises(RuntimeError, match="finite"):
        model.gatet(groups)


def test_positional_api_and_refit_replaces_identity(fitted):
    model, groups = deepcopy(fitted)
    assert model.estimate("ATE", 0.05, False).diagnostic_data is None
    model.data.df["id"] = "new_" + model.data.df["id"]
    model.fit()
    with pytest.raises(ValueError, match="fit-time"):
        model.gate(groups)
    groups.index = "new_" + groups.index
    assert np.isfinite(model.gate(groups).values).all()


def test_gate_and_gatet_target_different_populations(fitted):
    model, groups = deepcopy(fitted)
    # Supply oracle nuisances and outcomes: treatment effects vary with X,
    # and treatment recipients occupy systematically different X regions.
    x = np.linspace(-2, 2, len(model._y))
    labels = np.where(x < -0.5, 0, np.where(x < 0.5, 1, 2))
    model._d = np.eye(3, dtype=int)[labels]
    model.g_hat_ = np.column_stack([x, x + 2 + x, x + 3 + 2 * x])
    model._y = model.g_hat_[np.arange(len(x)), labels]
    groups[:] = "all"
    gate, gatet = model.gate(groups), model.gatet(groups)
    assert_allclose(gate.values[0], [np.mean(2 + x), np.mean(3 + 2 * x)])
    assert_allclose(
        gatet.values[0], [np.mean(2 + x[labels == 1]), np.mean(3 + 2 * x[labels == 2])]
    )
    assert abs(gate.values[0, 1] - gatet.values[0, 1]) > 1
