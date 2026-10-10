"""Joint subgroup effect results for multiple treatments versus one baseline."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .gate_estimate import GateEstimate


class MultiGateEstimate(BaseModel):
    """Group-by-treatment effects with pointwise inference and joint covariance.

    Effect arrays have shape ``(len(group_names), len(treatment_names))``.
    ``treatment_names`` contains only active arms. Covariance axes use the
    group-major ``(group, treatment)`` product. Joint covariance does not imply
    simultaneous confidence intervals or multiplicity-adjusted significance.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    estimand: str
    model: str = "MultiTreatmentIRM"
    group_names: List[str]
    treatment_names: List[str]
    baseline_treatment: str
    values: np.ndarray
    std_errors: np.ndarray
    test_stats: np.ndarray
    p_values: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    alpha: float
    covariance: pd.DataFrame
    summary_table: pd.DataFrame
    n_group: np.ndarray
    n_treated: np.ndarray
    n_control: np.ndarray
    n_other: np.ndarray
    model_options: Dict[str, Any] = Field(default_factory=dict)
    diagnostic_data: Optional[Dict[str, Any]] = None
    time: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    @model_validator(mode="after")
    def _validate_shapes(self) -> "MultiGateEstimate":
        for field in ("group_names", "treatment_names"):
            names = getattr(self, field)
            if not names or len(set(names)) != len(names):
                raise ValueError(f"{field} must be non-empty and unique.")
        if self.baseline_treatment in self.treatment_names:
            raise ValueError("treatment_names must contain only active arms.")
        if self.estimand not in {"GATE", "GATET"}:
            raise ValueError("estimand must be GATE or GATET.")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        shape = (len(self.group_names), len(self.treatment_names))
        for field in (
            "values",
            "std_errors",
            "test_stats",
            "p_values",
            "ci_lower",
            "ci_upper",
            "n_treated",
            "n_other",
        ):
            if np.asarray(getattr(self, field)).shape != shape:
                raise ValueError(f"{field} must have shape {shape}.")
        for field in ("n_group", "n_control"):
            if np.asarray(getattr(self, field)).shape != (shape[0],):
                raise ValueError(f"{field} must have shape ({shape[0]},).")
        index = pd.MultiIndex.from_product(
            [self.group_names, self.treatment_names], names=["group", "treatment"]
        )
        for axis in (
            self.covariance.index,
            self.covariance.columns,
            self.summary_table.index,
        ):
            if (
                not isinstance(axis, pd.MultiIndex)
                or not axis.is_unique
                or not axis.equals(index)
                or axis.names != index.names
            ):
                raise ValueError(
                    "Result tables must use the group-major (group, treatment) index."
                )
        self.summary_table = self.summary_table.copy()
        self.summary_table["is_significant"] = self.p_values.ravel() < self.alpha
        return self

    def summary(self) -> pd.DataFrame:
        """Return one row per group and active treatment, with pointwise inference."""
        return self.summary_table.reset_index().copy()

    def for_treatment(self, name: str) -> GateEstimate:
        """Extract an arm for existing subgroup contrasts and ``gate_plot``.

        For GATET, each arm targets its own recipients. Differences between
        arms' GATETs therefore also reflect different target populations.
        """
        if name not in self.treatment_names:
            raise ValueError(
                f"Unknown active treatment {name!r}; expected one of {self.treatment_names}."
            )
        arm = self.treatment_names.index(name)
        positions = np.arange(len(self.group_names)) * len(self.treatment_names) + arm
        table = self.summary_table.xs(name, level="treatment").copy()
        covariance = self.covariance.iloc[positions, positions].copy()
        covariance.index = covariance.columns = pd.Index(self.group_names, name="group")
        diagnostics = deepcopy(self.diagnostic_data)
        if diagnostics is not None:
            for field in (
                "orthogonal_signal",
                "raw_signal",
                "covariance_contributions",
            ):
                if field in diagnostics:
                    diagnostics[field] = diagnostics[field][:, arm].copy()
            diagnostics["group_diagnostics"] = table.copy()
            diagnostics["treatment"] = name
        return GateEstimate(
            estimand=self.estimand,
            model=self.model,
            group_names=list(self.group_names),
            **{
                field: getattr(self, field)[:, arm].copy()
                for field in (
                    "values",
                    "std_errors",
                    "test_stats",
                    "p_values",
                    "ci_lower",
                    "ci_upper",
                )
            },
            alpha=self.alpha,
            covariance=covariance,
            summary_table=table,
            n_group=self.n_group.copy(),
            n_treated=self.n_treated[:, arm].copy(),
            n_control=self.n_control.copy(),
            **{
                field: table[field].to_numpy(copy=True)
                for field in (
                    "share_treated",
                    "mean_phi",
                    "std_phi",
                    "mean_propensity",
                    "min_propensity",
                    "max_propensity",
                )
            },
            model_options={
                **deepcopy(self.model_options),
                "treatment": name,
                "baseline_treatment": self.baseline_treatment,
                "covariance_structure": "diagonal",
            },
            diagnostic_data=diagnostics,
            time=self.time,
        )
