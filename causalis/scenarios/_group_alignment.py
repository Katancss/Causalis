"""Fit-time observation alignment shared by binary and multi-treatment subgroups."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

_FIT_INDEX_SOURCE_USER_ID = "user_id"
_FIT_INDEX_SOURCE_ROW_INDEX = "row_index"


def _user_id_name(data: Any) -> Optional[str]:
    """Resolve the ID column for CausalData and MultiCausalData contracts."""
    if data is None:
        return None
    if hasattr(data, "user_id_name"):
        return data.user_id_name
    return getattr(data, "user_id", None)


def _groups_alignment_msg(estimand: str) -> str:
    return (
        f"{estimand} groups must be indexed by the fit-time observation ids, or by the original fit-time row index "
        "when that row-to-observation-id mapping is still unchanged. "
        "Fit the IRM with a stable DataFrame index or CausalData.user_id, and preserve that indexing on groups."
    )


def _is_default_positional_index(index: pd.Index, *, n_obs: int) -> bool:
    """Return whether an index is the default positional 0..n-1 labeling."""
    if len(index) != n_obs:
        return False
    if isinstance(index, pd.RangeIndex):
        return index.start == 0 and index.stop == n_obs and index.step == 1
    if not pd.api.types.is_integer_dtype(index.dtype):
        return False
    index_values = index.to_numpy(copy=False)
    return bool(np.array_equal(index_values, np.arange(n_obs, dtype=index_values.dtype)))


def _resolve_fit_observation_index(
    *,
    irm_model: Any,
    n_obs: int,
    estimand: str = "GATE",
) -> tuple[pd.Index, str]:
    """Resolve stable fit-time observation ids used to align group labels."""
    data_obj = getattr(irm_model, "data", None)
    user_id_name = _user_id_name(data_obj)
    fit_index = getattr(irm_model, "_fit_index_", None)
    if fit_index is not None:
        fit_index_resolved = pd.Index(fit_index).copy()
        fit_index_source = (
            _FIT_INDEX_SOURCE_USER_ID
            if getattr(irm_model, "_fit_row_index_", None) is not None or fit_index_resolved.name == user_id_name
            else _FIT_INDEX_SOURCE_ROW_INDEX
        )
    else:
        if data_obj is None:
            raise RuntimeError(
                f"IRM does not expose fit-time observation ids required for {estimand} group alignment. "
                "Refit the model with the current version."
            )
        if user_id_name:
            fit_index_resolved = pd.Index(data_obj.df[user_id_name].copy(), name=user_id_name)
            fit_index_source = _FIT_INDEX_SOURCE_USER_ID
        else:
            fit_index_resolved = pd.Index(data_obj.df.index.copy())
            fit_index_source = _FIT_INDEX_SOURCE_ROW_INDEX

    if len(fit_index_resolved) != n_obs:
        raise RuntimeError("Stored fit-time observation ids have inconsistent length; refit the model.")

    if fit_index_source == _FIT_INDEX_SOURCE_ROW_INDEX and _is_default_positional_index(fit_index_resolved, n_obs=n_obs):
        raise ValueError(
            f"{_groups_alignment_msg(estimand)} "
            "The fitted sample currently only has a default positional index (0..n-1), which is ambiguous."
        )

    return fit_index_resolved, fit_index_source


def _align_groups_to_fit_observations(
    groups: pd.DataFrame | pd.Series | np.ndarray,
    *,
    irm_model: Any,
    n_obs: int,
    estimand: str = "GATE",
) -> pd.DataFrame | pd.Series:
    """Align group labels to fit-time observation ids and reject ambiguous positional inputs."""
    if isinstance(groups, np.ndarray):
        # When groups is a raw numpy array, we assume it is already aligned with
        # the original fit-time row index. We wrap it in a Series using that index
        # so the downstream alignment logic can correctly map it to the fit-time ids.
        fit_row_index = getattr(irm_model, "_fit_row_index_", None)
        if fit_row_index is None:
            data_obj = getattr(irm_model, "data", None)
            fit_row_index = getattr(data_obj.df, "index", None) if data_obj is not None else None

        if fit_row_index is not None and len(fit_row_index) == n_obs:
            groups = pd.Series(groups, index=fit_row_index)
        else:
            # If we can't find a reliable row index, we wrap it with the fit_index directly
            # assuming the user knows what they are doing (alignment by position).
            fit_index, _ = _resolve_fit_observation_index(irm_model=irm_model, n_obs=n_obs, estimand=estimand)
            groups = pd.Series(groups, index=fit_index)

    if isinstance(groups, pd.Series):
        groups_df = groups.to_frame()
        original_is_series = True
        original_name = groups.name
    elif isinstance(groups, pd.DataFrame):
        groups_df = groups.copy()
        original_is_series = False
        original_name = None
    else:
        raise TypeError("groups must be a pandas Series, DataFrame or numpy array.")

    fit_index, _ = _resolve_fit_observation_index(irm_model=irm_model, n_obs=n_obs, estimand=estimand)

    if groups_df.shape[0] != n_obs:
        if fit_index.is_unique and groups_df.index.is_unique and set(fit_index).issubset(set(groups_df.index)):
            aligned = groups_df.reindex(fit_index)
            if original_is_series:
                return aligned.iloc[:, 0].rename(original_name)
            return aligned

        aligned = _align_groups_via_fit_row_index(groups_df=groups_df, irm_model=irm_model, fit_index=fit_index)
        if aligned is not None:
            if original_is_series:
                return aligned.iloc[:, 0].rename(original_name)
            return aligned

        raise ValueError(f"groups must align to the {n_obs} retained fit observations.")

    if not groups_df.index.is_unique:
        raise ValueError("groups index must be unique so rows can be aligned to the fit-time sample.")

    if groups_df.index.equals(fit_index):
        aligned = groups_df
    elif fit_index.is_unique and set(groups_df.index) == set(fit_index):
        aligned = groups_df.reindex(fit_index)
    else:
        aligned = _align_groups_via_fit_row_index(groups_df=groups_df, irm_model=irm_model, fit_index=fit_index)
        if aligned is None:
            raise ValueError(
                f"{_groups_alignment_msg(estimand)} "
                "groups.index must either match the fit-time ids (or a permutation of them), "
                "or match the original fit-time row index while the current row-to-id mapping remains unchanged."
            )

    if original_is_series:
        return aligned.iloc[:, 0].rename(original_name)
    return aligned


def _align_groups_via_fit_row_index(
    *,
    groups_df: pd.DataFrame,
    irm_model: Any,
    fit_index: pd.Index,
) -> Optional[pd.DataFrame]:
    """Map row-indexed groups to fit-time observation ids when the row->id mapping is unchanged."""
    fit_row_index = getattr(irm_model, "_fit_row_index_", None)
    data_obj = getattr(irm_model, "data", None)
    user_id_name = _user_id_name(data_obj)

    if fit_row_index is None or data_obj is None or user_id_name is None:
        return None

    fit_row_index_resolved = pd.Index(fit_row_index).copy()
    if len(fit_row_index_resolved) != len(fit_index):
        raise RuntimeError("Stored fit-time row index has inconsistent length; refit the model.")
    if not fit_row_index_resolved.is_unique:
        return None

    if groups_df.index.equals(fit_row_index_resolved):
        groups_by_row = groups_df
    elif set(fit_row_index_resolved).issubset(set(groups_df.index)):
        groups_by_row = groups_df.reindex(fit_row_index_resolved)
    elif set(groups_df.index) == set(fit_row_index_resolved):
        groups_by_row = groups_df.reindex(fit_row_index_resolved)
    else:
        return None

    current_row_index = pd.Index(data_obj.df.index.copy())
    current_fit_ids = pd.Index(data_obj.df[user_id_name].copy(), name=user_id_name)
    if len(current_row_index) != len(fit_index) or len(current_fit_ids) != len(fit_index):
        return None
    if not current_row_index.is_unique:
        return None

    fit_mapping = pd.Series(np.asarray(fit_index, dtype=object), index=fit_row_index_resolved)
    current_mapping = pd.Series(np.asarray(current_fit_ids, dtype=object), index=current_row_index)
    if not current_mapping.reindex(fit_row_index_resolved).equals(fit_mapping):
        return None

    aligned = groups_by_row.copy()
    aligned.index = fit_index
    return aligned


