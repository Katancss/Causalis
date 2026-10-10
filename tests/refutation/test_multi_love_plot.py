from __future__ import annotations

import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from causalis.data_contracts.causal_diagnostic_data import MultiUnconfoundednessDiagnosticData
from causalis.data_contracts.multicausal_estimate import MultiCausalEstimate
from causalis.data_contracts.multicausaldata import MultiCausalData
from causalis.scenarios.multi_unconfoundedness.refutation import love_plot
from causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness import love_plot as nested_love_plot

validation = importlib.import_module(
    "causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.unconfoundedness_validation"
)
plot_module = importlib.import_module(
    "causalis.scenarios.multi_unconfoundedness.refutation.unconfoundedness.love_plot"
)


def make_inputs(n=120, p=4, k=3, score="ATE", normalize=False):
    rng = np.random.default_rng(57)
    labels = rng.choice(k, size=n)
    labels[:k] = np.arange(k)
    x = rng.normal(size=(n, p)) + labels[:, None] * 0.3
    d = np.eye(k)[labels]
    m = rng.dirichlet(np.ones(k), size=n)
    return validation._BalanceInputs(
        x, d, m, [f"x{i}" for i in range(p)], [f"d_{i}" for i in range(k)], score, normalize
    )


def make_data_estimate(**kwargs):
    inputs = make_inputs(**kwargs)
    frame = pd.DataFrame(inputs.x, columns=inputs.feature_names)
    for i, name in enumerate(inputs.treatment_names):
        frame[name] = inputs.d[:, i]
    frame["y"] = np.arange(len(frame), dtype=float)
    data = MultiCausalData(
        df=frame, outcome="y", confounders=inputs.feature_names,
        treatment_names=inputs.treatment_names, control_treatment=inputs.treatment_names[0],
    )
    diag = MultiUnconfoundednessDiagnosticData(
        x=inputs.x, d=inputs.d, m_hat=inputs.m_hat,
        score=inputs.score, normalize_ipw=inputs.normalize,
    )
    zeros = np.zeros(inputs.d.shape[1] - 1)
    estimate = MultiCausalEstimate(
        estimand=inputs.score, model="MultiTreatmentIRM", model_options={},
        value=zeros, ci_lower_absolute=zeros, ci_upper_absolute=zeros, alpha=0.05,
        p_value=np.ones_like(zeros), is_significant=[False] * len(zeros),
        n_treated=int(inputs.d[:, 1:].sum()), n_control=int(inputs.d[:, 0].sum()),
        outcome="y", treatment=inputs.treatment_names, confounders=inputs.feature_names,
        contrast_labels=[f"{name} vs d_0" for name in inputs.treatment_names[1:]],
        diagnostic_data=diag,
    )
    return data, estimate


def reference_smd(inputs):
    """Original full-array calculation, independent of the new moment helper."""
    x, d = inputs.x, inputs.d
    m = np.clip(inputs.m_hat, 1e-12, 1 - 1e-12)
    before, after = [], []
    for t in range(1, d.shape[1]):
        wb = d[:, 0] / m[:, 0] if inputs.score == "ATE" else d[:, 0] * m[:, t] / m[:, 0]
        wt = d[:, t] / m[:, t] if inputs.score == "ATE" else d[:, t]
        if inputs.normalize:
            wb = wb / wb.mean()
            wt = wt / wt.mean()
        mb = (wb[:, None] * x).sum(axis=0) / wb.sum()
        mt = (wt[:, None] * x).sum(axis=0) / wt.sum()
        vb = (wb[:, None] * (x - mb) ** 2).sum(axis=0) / wb.sum()
        vt = (wt[:, None] * (x - mt) ** 2).sum(axis=0) / wt.sum()
        after.append(validation._smd_from_moments(mu_a=mt, var_a=vt, mu_b=mb, var_b=vb))
        xb, xt = x[d[:, 0].astype(bool)], x[d[:, t].astype(bool)]
        before.append(validation._smd_from_moments(
            mu_a=xt.mean(axis=0), var_a=xt.var(axis=0),
            mu_b=xb.mean(axis=0), var_b=xb.var(axis=0),
        ))
    return np.array(before).T, np.array(after).T


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("offset", [0.0, 1e8])
def test_balance_matches_original_formulas(score, normalize, offset, monkeypatch):
    inputs = make_inputs(n=401, k=5, score=score, normalize=normalize)
    inputs.x += offset
    original = [a.copy() for a in (inputs.x, inputs.d, inputs.m_hat)]
    monkeypatch.setattr(validation, "_MOMENT_WORK_BYTES", 8 * 4 * 17)
    before, after = reference_smd(inputs)
    balance = validation._balance_smd(inputs, threshold=0.1)
    np.testing.assert_allclose(balance["smd"], after, rtol=1e-6, atol=2e-7 if offset else 1e-12)
    np.testing.assert_allclose(balance["smd_unweighted"], before, rtol=1e-6, atol=2e-7 if offset else 1e-12)
    for array, saved in zip((inputs.x, inputs.d, inputs.m_hat), original):
        np.testing.assert_array_equal(array, saved)


def test_constant_features_and_zero_mass():
    inputs = make_inputs()
    inputs.x[:, 0] = 3
    inputs.x[:, 1] = inputs.d[:, 1] * 2
    balance = validation._balance_smd(inputs, threshold=0.1)
    assert (balance["smd"].iloc[0] == 0).all()
    assert np.isinf(balance["smd"].iloc[1, 0])
    inputs.d[:, 1] = 0
    with pytest.raises(RuntimeError, match="zero total mass"):
        validation._balance_smd(inputs, threshold=0.1)


def test_moment_buffer_is_bounded_and_reused(monkeypatch):
    inputs = make_inputs(n=400, p=7)
    budget = 8 * 7 * 13
    monkeypatch.setattr(validation, "_MOMENT_WORK_BYTES", budget)
    original_empty = np.empty
    allocations = []
    def tracked_empty(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        allocations.append(result.nbytes)
        return result
    monkeypatch.setattr(validation.np, "empty", tracked_empty)
    validation._weighted_moments(inputs.x, inputs.d[:, 0], inputs.d[:, 0].sum())
    assert allocations == [budget]


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
@pytest.mark.parametrize("k", [2, 3, 5])
def test_plot_values_layout_and_cleanup(score, k, monkeypatch):
    data, estimate = make_data_estimate(score=score, k=k)
    report = validation.run_unconfoundedness_diagnostics(data, estimate, return_summary=False)
    calls = []
    def diagnostics(**kwargs):
        calls.append(kwargs)
        return report
    monkeypatch.setattr(plot_module, "run_unconfoundedness_diagnostics", diagnostics)
    registered = plt.get_fignums()
    fig = love_plot(data, estimate, dpi=60)
    assert plt.get_fignums() == registered
    assert len(calls) == 1 and calls[0]["return_summary"] is False
    assert len(fig.axes) == k - 1
    before, after = report["balance"]["smd_unweighted"], report["balance"]["smd"]
    order = np.argsort(-np.fmax(before.max(axis=1), after.max(axis=1)), kind="stable")
    for idx, ax in enumerate(fig.axes):
        assert before.columns[idx] in ax.get_title()
        assert score in ax.get_title()
        assert [t.get_text() for t in ax.get_yticklabels()] == list(before.index[order])
        assert ax.get_xlim() == fig.axes[0].get_xlim()
        assert ax.yaxis_inverted()
        points = [c for c in ax.collections if isinstance(c, PathCollection)]
        np.testing.assert_allclose(points[0].get_offsets()[:, 0], before.iloc[order, idx])
        np.testing.assert_allclose(points[1].get_offsets()[:, 0], after.iloc[order, idx])
        assert sum(isinstance(c, LineCollection) for c in ax.collections) == 1
        assert len(ax.lines) == 1
        np.testing.assert_allclose(ax.lines[0].get_xdata(), [0.1, 0.1])
        assert len(ax.get_legend().get_texts()) == 3
    assert nested_love_plot is love_plot


def test_fallback_does_not_copy_whole_dataframe(monkeypatch):
    data, estimate = make_data_estimate()
    estimate.diagnostic_data.x = None
    def unexpected_copy(*args, **kwargs):
        pytest.fail("get_df must not copy unrelated columns")
    monkeypatch.setattr(MultiCausalData, "get_df", unexpected_copy)
    fig = love_plot(data, estimate, dpi=60)
    assert len(fig.axes) == 2


def test_nonfinite_values_and_stable_order(monkeypatch):
    data, estimate = make_data_estimate()
    report = validation.run_unconfoundedness_diagnostics(data, estimate, return_summary=False)
    for key in ("smd", "smd_unweighted"):
        report["balance"][key].iloc[:, :] = [[0.2, 0.2], [np.inf, 0.1], [0.2, 0.2], [np.nan, np.nan]]
    monkeypatch.setattr(plot_module, "run_unconfoundedness_diagnostics", lambda **kw: report)
    fig = love_plot(data, estimate, dpi=60)
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["x1", "x0", "x2", "x3"]
    assert len(fig.axes[0].texts) == 1 and not fig.axes[1].texts
    for ax in fig.axes:
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                assert len(collection.get_offsets()) == 3
                assert np.isfinite(collection.get_offsets()).all()


def test_save_options_and_failure_cleanup(tmp_path, monkeypatch):
    data, estimate = make_data_estimate()
    output = tmp_path / "balance.png"
    fig = love_plot(data, estimate, figsize=(12, 6), dpi=60, save=output, save_dpi=72, transparent=True)
    assert output.stat().st_size > 0
    np.testing.assert_allclose(fig.get_size_inches(), [12, 6])
    assert fig.dpi == 60
    def fail(*args, **kwargs):
        raise OSError("save failed")
    monkeypatch.setattr(Figure, "savefig", fail)
    registered = plt.get_fignums()
    with pytest.raises(OSError, match="save failed"):
        love_plot(data, estimate, dpi=60, save=output)
    assert plt.get_fignums() == registered


def test_errors_and_repeated_calls():
    data, estimate = make_data_estimate()
    with pytest.raises(ValueError, match="diagnostic_data"):
        love_plot(data, estimate.model_copy(update={"diagnostic_data": None}))
    with pytest.raises(ValueError, match="estimate.outcome"):
        love_plot(data, estimate.model_copy(update={"outcome": "other"}))
    with pytest.raises(TypeError, match="MultiCausalData"):
        love_plot(data.df, estimate)
    registered = plt.get_fignums()
    for _ in range(3):
        love_plot(data, estimate, dpi=60)
    assert plt.get_fignums() == registered


def test_large_feature_set_autoscales():
    data, estimate = make_data_estimate(p=65)
    fig = love_plot(data, estimate, dpi=40)
    assert fig.get_size_inches()[1] > 15
    assert all(len(ax.get_yticklabels()) == 65 for ax in fig.axes)


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
def test_fitted_multi_treatment_irm(score):
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LogisticRegression
    from causalis.scenarios.multi_unconfoundedness import MultiTreatmentIRM

    data, _ = make_data_estimate()
    model = MultiTreatmentIRM(
        data=data, ml_g=DummyRegressor(), ml_m=LogisticRegression(),
        n_folds=2, random_state=42,
    ).fit()
    estimate = model.estimate(score=score, diagnostic_data=True)
    fig = love_plot(data, estimate, dpi=60)
    assert len(fig.axes) == 2
    assert all(score in ax.get_title() for ax in fig.axes)
