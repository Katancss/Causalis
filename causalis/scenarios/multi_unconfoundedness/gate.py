"""Post-estimation subgroup effects from fitted multiclass nuisance predictions."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from causalis.data_contracts.multi_gate_estimate import MultiGateEstimate
from causalis.scenarios._group_alignment import _align_groups_to_fit_observations
from causalis.scenarios.gate.model import _coerce_groups_to_partition


def _estimate_multi_gate(
    model: Any,
    *,
    score: str,
    groups: pd.Series | pd.DataFrame | np.ndarray | None,
    alpha: float,
    cov_type: str,
    diagnostic_data: bool,
) -> MultiGateEstimate:
    """Estimate canonical subgroup moments without changing model state.

    GATE averages arm-versus-baseline AIPW signals over all subgroup members.
    GATET divides the subgroup ATT numerator by the number receiving that arm.
    The covariance is the Gram matrix of scaled moment residuals, with the
    same marginal HC corrections as the binary subgroup estimators.
    """
    cov_type = str(cov_type).upper()
    if cov_type not in {"HC0", "HC1", "HC2", "HC3"}:
        raise ValueError("cov_type must be one of {HC0, HC1, HC2, HC3}.")
    if groups is None:
        raise ValueError(
            f"{score} requires explicit groups=... defining a pre-treatment partition."
        )
    if not getattr(model, "_fit_user_id_name_", None):
        raise ValueError(
            f"{score} requires MultiCausalData.user_id at fit time; configure it and refit."
        )
    if getattr(model, "_fit_index_", None) is None:
        raise RuntimeError("Fit-time observation IDs are unavailable; refit the model.")

    y, d = model._resolve_estimation_targets()
    n, n_arms = d.shape
    mu, e = np.asarray(model.g_hat_), np.asarray(model.m_hat_)
    if (
        mu.shape != d.shape
        or e.shape != d.shape
        or not (np.isfinite(y).all() and np.isfinite(mu).all() and np.isfinite(e).all())
    ):
        raise RuntimeError(
            "Fitted outcomes and nuisance predictions must be finite and have matching shapes."
        )
    propensity_denominators = e if score == "GATE" else e[:, [0]]
    if np.any(propensity_denominators <= np.finfo(float).tiny):
        raise RuntimeError(
            "Required propensity denominators are numerically zero; increase trimming_threshold and refit."
        )
    aligned = _align_groups_to_fit_observations(
        groups, irm_model=model, n_obs=n, estimand=score
    )
    partition = _coerce_groups_to_partition(aligned, n_obs=n)
    names, codes = partition.group_names, partition.codes
    if len(set(names)) != len(names):
        raise ValueError("Group labels must have unique string representations.")
    treatments = list(model._fit_treatment_names_)
    baseline, active = treatments[0], treatments[1:]
    n_groups, n_active = len(names), n_arms - 1
    n_group = np.bincount(codes, minlength=n_groups)
    counts = np.zeros((n_groups, n_arms), dtype=int)
    np.add.at(counts, codes, d)
    n_control, n_treated = counts[:, 0], counts[:, 1:]
    n_other = n_group[:, None] - n_control[:, None] - n_treated
    invalid = n_treated == 0
    if score == "GATE":
        invalid |= n_control[:, None] == 0
    if invalid.any():
        cells = [(names[g], active[k]) for g, k in np.argwhere(invalid)]
        support = "baseline and active-arm" if score == "GATE" else "active-arm"
        raise ValueError(
            f"{score} requires {support} observations in every group. Invalid cells: {cells}."
        )

    warning_messages: list[str] = []

    def warn(message: str) -> None:
        warning_messages.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    if score == "GATET" and np.any(n_control == 0):
        missing = [names[g] for g in np.flatnonzero(n_control == 0)]
        warn(
            f"GATET groups {missing} have no baseline observations; within-group overlap is degenerate."
        )
    if model.normalize_ipw:
        warn(
            f"normalize_ipw=True is ignored for {score} to preserve canonical unnormalized orthogonal signals."
        )

    residual0 = y[:, None] - mu[:, [0]]
    if score == "GATE":
        signal = (
            mu[:, 1:]
            - mu[:, [0]]
            + d[:, 1:] / e[:, 1:] * (y[:, None] - mu[:, 1:])
            - d[:, [0]] / e[:, [0]] * residual0
        )
        denominator = np.broadcast_to(n_group[:, None], (n_groups, n_active))
    else:
        signal = (d[:, 1:] - d[:, [0]] * e[:, 1:] / e[:, [0]]) * residual0
        denominator = n_treated
    if not np.isfinite(signal).all():
        raise RuntimeError(f"Computed {score} signal contains non-finite values.")
    totals = np.zeros((n_groups, n_active))
    np.add.at(totals, codes, signal)
    values = totals / denominator
    if not np.isfinite(values).all():
        raise RuntimeError(f"Computed {score} estimates contain non-finite values.")
    share = n_treated / n_group[:, None]
    phi = signal if score == "GATE" else signal / share[codes]
    residual = signal - (values[codes] if score == "GATE" else d[:, 1:] * values[codes])
    contributions = residual / denominator[codes]

    valid = np.broadcast_to(n_group[:, None] > 1, values.shape).copy()
    if score == "GATET" and cov_type in {"HC2", "HC3"}:
        valid &= n_treated > 1
    if not valid.all():
        cells = [(names[g], active[k]) for g, k in np.argwhere(~valid)]
        warn(
            f"{score} {cov_type} inference is undefined for singleton groups or treated leverage 1; "
            f"inference is set to NaN for cells {cells}."
        )
    if cov_type == "HC1":
        if n <= n_groups:
            warn(
                "HC1 scaling n/(n-n_groups) is undefined; falling back to HC0 scaling."
            )
        else:
            contributions *= np.sqrt(n / (n - n_groups))
    elif cov_type in {"HC2", "HC3"}:
        leverage = (
            1 / denominator[codes] if score == "GATE" else d[:, 1:] / denominator[codes]
        )
        correction = np.sqrt(1 - leverage) if cov_type == "HC2" else 1 - leverage
        contributions = np.divide(
            contributions,
            correction,
            out=np.zeros_like(contributions),
            where=correction > 0,
        )
    contributions = np.where(valid[codes], contributions, 0.0)

    # Disjoint groups give separate covariance blocks. Avoid an n x (G*K)
    # dense design; within each group, retain the full cross-arm Gram matrix.
    covariance = np.zeros((n_groups * n_active, n_groups * n_active))
    rows_by_group = np.split(np.argsort(codes, kind="stable"), np.cumsum(n_group)[:-1])
    mean_phi = np.empty_like(values)
    std_phi = np.full_like(values, np.nan)
    propensity_mean = np.empty((n_groups, n_arms))
    propensity_min = np.empty_like(propensity_mean)
    propensity_max = np.empty_like(propensity_mean)
    for g, rows in enumerate(rows_by_group):
        block = slice(g * n_active, (g + 1) * n_active)
        covariance[block, block] = contributions[rows].T @ contributions[rows]
        mean_phi[g] = phi[rows].mean(axis=0)
        if n_group[g] > 1:
            std_phi[g] = phi[rows].std(axis=0, ddof=1)
        propensity_mean[g] = e[rows].mean(axis=0)
        propensity_min[g] = e[rows].min(axis=0)
        propensity_max[g] = e[rows].max(axis=0)
        if n_group[g] < 10:
            warning_messages.append(
                f"{score} group {names[g]!r} has small support (n={n_group[g]})."
            )
        if np.any(propensity_min[g] < 0.05) or np.any(propensity_max[g] > 0.95):
            warning_messages.append(
                f"{score} group {names[g]!r} has extreme multiclass propensity support."
            )
    if not np.isfinite(covariance).all():
        raise RuntimeError(
            f"Computed {score} covariance is non-finite; inspect score stability."
        )
    covariance[~valid.ravel(), :] = np.nan
    covariance[:, ~valid.ravel()] = np.nan
    std_errors = np.sqrt(np.diag(covariance).reshape(values.shape))
    with np.errstate(divide="ignore", invalid="ignore"):
        test_stats = values / std_errors
    p_values = 2 * norm.sf(np.abs(test_stats))
    critical = norm.ppf(1 - alpha / 2)
    lower, upper = values - critical * std_errors, values + critical * std_errors
    index = pd.MultiIndex.from_product([names, active], names=["group", "treatment"])
    table = pd.DataFrame(
        {
            "baseline_treatment": baseline,
            "value": values.ravel(),
            "std_error": std_errors.ravel(),
            "test_stat": test_stats.ravel(),
            "p_value": p_values.ravel(),
            "is_significant": p_values.ravel() < alpha,
            "ci_lower": lower.ravel(),
            "ci_upper": upper.ravel(),
            "n_group": np.repeat(n_group, n_active),
            "n_treated": n_treated.ravel(),
            "n_control": np.repeat(n_control, n_active),
            "n_other": n_other.ravel(),
            "share_treated": share.ravel(),
            "mean_phi": mean_phi.ravel(),
            "std_phi": std_phi.ravel(),
            "mean_propensity": propensity_mean[:, 1:].ravel(),
            "min_propensity": propensity_min[:, 1:].ravel(),
            "max_propensity": propensity_max[:, 1:].ravel(),
            "mean_baseline_propensity": np.repeat(propensity_mean[:, 0], n_active),
            "min_baseline_propensity": np.repeat(propensity_min[:, 0], n_active),
            "max_baseline_propensity": np.repeat(propensity_max[:, 0], n_active),
        },
        index=index,
    )
    options = {
        "cov_type": cov_type,
        "covariance_structure": "block_diagonal_by_group",
        "inference": "pointwise",
        "normalize_ipw_requested": bool(model.normalize_ipw),
        "normalize_ipw_effective": False,
        "trimming_rule": model.trimming_rule,
        "trimming_threshold": model.trimming_threshold,
        "n_folds": model.n_folds,
        "n_rep": model.n_rep,
        "random_state": model.random_state,
        "group_representation": "compact_codes",
        "group_warning_messages": warning_messages,
    }
    diagnostics = None
    if diagnostic_data and getattr(model, "_fit_store_diagnostics_", False):
        diagnostics = {
            "orthogonal_signal": phi.copy(),
            "raw_signal": signal.copy(),
            # Each row contains the columns for its own group; other groups' entries are zero.
            "covariance_contributions": np.where(valid[codes], contributions, np.nan),
            "group_codes": codes.copy(),
            "group_names": list(names),
            "group_diagnostics": table.copy(),
            "group_warning_messages": list(warning_messages),
            "observation_ids": model._fit_index_.copy(),
        }
    return MultiGateEstimate(
        estimand=score,
        group_names=names,
        treatment_names=active,
        baseline_treatment=baseline,
        values=values,
        std_errors=std_errors,
        test_stats=test_stats,
        p_values=p_values,
        ci_lower=lower,
        ci_upper=upper,
        alpha=alpha,
        covariance=pd.DataFrame(covariance, index=index, columns=index),
        summary_table=table,
        n_group=n_group,
        n_treated=n_treated,
        n_control=n_control,
        n_other=n_other,
        model_options=options,
        diagnostic_data=diagnostics,
    )
