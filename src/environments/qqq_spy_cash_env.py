"""Framework-agnostic QQQ/SPY/cash portfolio accounting environment.

For each step, old holdings earn the close-to-next-open overnight return, the
portfolio rebalances at the next regular-session open, and new holdings earn
that session's open-to-close return.  This matches dataset version
``qqq_spy_rl_daily_v1`` and prevents same-close execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Sequence

import numpy as np
import pandas as pd


ASSET_NAMES = ("QQQ", "SPY", "CASH")

# Complete 25-percentage-point simplex, symmetric between QQQ and SPY.
DISCRETE_ACTION_WEIGHTS = np.asarray(
    [
        (0.00, 0.00, 1.00),
        (0.00, 0.25, 0.75),
        (0.00, 0.50, 0.50),
        (0.00, 0.75, 0.25),
        (0.00, 1.00, 0.00),
        (0.25, 0.00, 0.75),
        (0.25, 0.25, 0.50),
        (0.25, 0.50, 0.25),
        (0.25, 0.75, 0.00),
        (0.50, 0.00, 0.50),
        (0.50, 0.25, 0.25),
        (0.50, 0.50, 0.00),
        (0.75, 0.00, 0.25),
        (0.75, 0.25, 0.00),
        (1.00, 0.00, 0.00),
    ],
    dtype=np.float64,
)
DISCRETE_ACTION_WEIGHTS.setflags(write=False)


@dataclass(frozen=True)
class RewardConfig:
    """Explicit, provisional coefficients for the semiannual score."""

    partial_credit: float = 0.25
    drawdown_penalty: float = 0.0
    turnover_regularizer: float = 0.0
    score_clip: float | None = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.partial_credit <= 1.0:
            raise ValueError("partial_credit must be between 0 and 1")
        if self.drawdown_penalty < 0.0:
            raise ValueError("drawdown_penalty must be non-negative")
        if self.turnover_regularizer < 0.0:
            raise ValueError("turnover_regularizer must be non-negative")
        if self.score_clip is not None and self.score_clip <= 0.0:
            raise ValueError("score_clip must be positive or None")


def relative_performance_score(
    portfolio_return: float,
    spy_return: float,
    qqq_return: float,
    *,
    partial_credit: float = 0.25,
) -> tuple[float, str, float, float]:
    """Return ``score, tier, alpha_strong, alpha_weak``."""

    if not 0.0 <= partial_credit <= 1.0:
        raise ValueError("partial_credit must be between 0 and 1")

    stronger = max(spy_return, qqq_return)
    weaker = min(spy_return, qqq_return)
    alpha_strong = portfolio_return - stronger
    alpha_weak = portfolio_return - weaker

    if alpha_strong > 0.0:
        tier = "beat_both"
        score = alpha_strong + partial_credit * alpha_weak
    elif alpha_weak > 0.0:
        tier = "beat_one"
        score = partial_credit * alpha_weak
    else:
        tier = "beat_neither"
        score = alpha_weak
    return float(score), tier, float(alpha_strong), float(alpha_weak)


class QQQSpyCashEnv:
    """Long-only accounting core for one complete half-year episode."""

    def __init__(
        self,
        *,
        features: np.ndarray,
        overnight_returns: np.ndarray,
        intraday_returns: np.ndarray,
        feature_dates: Sequence[date | datetime | np.datetime64 | str],
        target_dates: Sequence[date | datetime | np.datetime64 | str],
        cash_overnight_returns: np.ndarray | None = None,
        cash_intraday_returns: np.ndarray | None = None,
        transaction_cost_rate: float = 0.001,
        initial_weights: np.ndarray | None = None,
        reward_config: RewardConfig | None = None,
        require_single_half_year: bool = True,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float64)
        self.overnight_returns = np.asarray(
            overnight_returns, dtype=np.float64
        )
        self.intraday_returns = np.asarray(
            intraday_returns, dtype=np.float64
        )
        self.feature_dates = np.asarray(
            feature_dates, dtype="datetime64[ns]"
        )
        self.target_dates = np.asarray(
            target_dates, dtype="datetime64[ns]"
        )
        n_steps = self.features.shape[0] if self.features.ndim == 2 else 0
        self.cash_overnight_returns = (
            np.zeros(n_steps, dtype=np.float64)
            if cash_overnight_returns is None
            else np.asarray(cash_overnight_returns, dtype=np.float64)
        )
        self.cash_intraday_returns = (
            np.zeros(n_steps, dtype=np.float64)
            if cash_intraday_returns is None
            else np.asarray(cash_intraday_returns, dtype=np.float64)
        )
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.initial_weights = (
            np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
            if initial_weights is None
            else np.asarray(initial_weights, dtype=np.float64)
        )
        self.reward_config = reward_config or RewardConfig()
        self.require_single_half_year = require_single_half_year
        self._validate_inputs()

        self.n_steps = self.features.shape[0]
        self.n_actions = int(DISCRETE_ACTION_WEIGHTS.shape[0])
        self.observation_size = int(self.features.shape[1] + 11)

        self._step_index = 0
        self._terminated = False
        self._weights = self.initial_weights.copy()
        self._portfolio_value = 1.0
        self._spy_value = 1.0
        self._qqq_value = 1.0
        self._peak_value = 1.0
        self._max_drawdown = 0.0
        self._cumulative_turnover = 0.0
        self._cumulative_transaction_cost = 0.0
        self._potential = 0.0

    @classmethod
    def from_dataset(
        cls,
        episode: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        **kwargs: Any,
    ) -> "QQQSpyCashEnv":
        """Construct one environment from one episode DataFrame."""

        if episode.empty:
            raise ValueError("episode cannot be empty")
        if "episode_id" in episode.columns:
            episode_ids = episode["episode_id"].unique()
            if len(episode_ids) != 1:
                raise ValueError("episode frame must contain one episode_id")
        missing_features = [
            column
            for column in feature_columns
            if column not in episode.columns
        ]
        if missing_features:
            raise ValueError(
                f"episode is missing feature columns: {missing_features}"
            )

        return cls(
            features=episode.loc[:, feature_columns].to_numpy(
                dtype=np.float64
            ),
            overnight_returns=episode.loc[
                :, ["qqq_overnight_return", "spy_overnight_return"]
            ].to_numpy(dtype=np.float64),
            intraday_returns=episode.loc[
                :, ["qqq_intraday_return", "spy_intraday_return"]
            ].to_numpy(dtype=np.float64),
            feature_dates=episode["feature_date"].to_numpy(),
            target_dates=episode["target_date"].to_numpy(),
            cash_overnight_returns=episode[
                "cash_overnight_return"
            ].to_numpy(dtype=np.float64),
            cash_intraday_returns=episode[
                "cash_intraday_return"
            ].to_numpy(dtype=np.float64),
            **kwargs,
        )

    def _validate_inputs(self) -> None:
        if self.features.ndim != 2 or self.features.shape[1] == 0:
            raise ValueError("features must be a non-empty 2D array")
        n_steps = self.features.shape[0]
        if n_steps < 1:
            raise ValueError("at least one step is required")
        for name, values, shape in (
            ("overnight_returns", self.overnight_returns, (n_steps, 2)),
            ("intraday_returns", self.intraday_returns, (n_steps, 2)),
            (
                "cash_overnight_returns",
                self.cash_overnight_returns,
                (n_steps,),
            ),
            (
                "cash_intraday_returns",
                self.cash_intraday_returns,
                (n_steps,),
            ),
            ("feature_dates", self.feature_dates, (n_steps,)),
            ("target_dates", self.target_dates, (n_steps,)),
        ):
            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")

        numeric_arrays = (
            ("features", self.features),
            ("overnight_returns", self.overnight_returns),
            ("intraday_returns", self.intraday_returns),
            ("cash_overnight_returns", self.cash_overnight_returns),
            ("cash_intraday_returns", self.cash_intraday_returns),
        )
        for name, values in numeric_arrays:
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
        for name, returns in numeric_arrays[1:]:
            if np.any(returns <= -1.0):
                raise ValueError(f"{name} must be greater than -100%")

        if np.any(np.isnat(self.feature_dates)) or np.any(
            np.isnat(self.target_dates)
        ):
            raise ValueError("dates cannot be missing")
        if np.any(
            np.diff(self.feature_dates) <= np.timedelta64(0, "ns")
        ) or np.any(np.diff(self.target_dates) <= np.timedelta64(0, "ns")):
            raise ValueError("feature_dates and target_dates must increase")
        if np.any(self.feature_dates >= self.target_dates):
            raise ValueError("each feature_date must precede target_date")

        if self.require_single_half_year:
            half_years = {
                self._half_year_key(value) for value in self.target_dates
            }
            if len(half_years) != 1:
                raise ValueError(
                    "all target_dates must belong to one calendar half-year"
                )

        if not 0.0 <= self.transaction_cost_rate < 1.0:
            raise ValueError(
                "transaction_cost_rate must be in the interval [0, 1)"
            )
        self._validate_weights(self.initial_weights, "initial_weights")

    @staticmethod
    def _half_year_key(value: np.datetime64) -> tuple[int, int]:
        day = value.astype("datetime64[D]").astype(object)
        return day.year, 1 if day.month <= 6 else 2

    @staticmethod
    def _validate_weights(weights: np.ndarray, label: str) -> None:
        if weights.shape != (3,):
            raise ValueError(f"{label} must have exactly three weights")
        if not np.all(np.isfinite(weights)):
            raise ValueError(f"{label} must contain only finite values")
        if np.any(weights < -1e-12):
            raise ValueError(f"{label} cannot contain negative weights")
        if not np.isclose(float(weights.sum()), 1.0, atol=1e-10):
            raise ValueError(f"{label} must sum to 1")

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to the configured initial book, normally 100% cash."""

        self._step_index = 0
        self._terminated = False
        self._weights = self.initial_weights.copy()
        self._portfolio_value = 1.0
        self._spy_value = 1.0
        self._qqq_value = 1.0
        self._peak_value = 1.0
        self._max_drawdown = 0.0
        self._cumulative_turnover = 0.0
        self._cumulative_transaction_cost = 0.0
        self._potential = 0.0
        return self._observation(), self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance old weights overnight, rebalance, then advance intraday."""

        if self._terminated:
            raise RuntimeError("step() called after the episode terminated")
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer action index")
        if action < 0 or action >= self.n_actions:
            raise ValueError(
                f"action must be between 0 and {self.n_actions - 1}"
            )

        step_index = self._step_index
        feature_date = self.feature_dates[step_index]
        target_date = self.target_dates[step_index]
        target_weights = DISCRETE_ACTION_WEIGHTS[int(action)].copy()

        overnight_components = np.asarray(
            (
                *self.overnight_returns[step_index],
                self.cash_overnight_returns[step_index],
            ),
            dtype=np.float64,
        )
        overnight_portfolio_return = float(
            np.dot(self._weights, overnight_components)
        )
        overnight_growth = 1.0 + overnight_portfolio_return
        self._portfolio_value *= overnight_growth
        self._weights = (
            self._weights * (1.0 + overnight_components) / overnight_growth
        )
        self._weights /= self._weights.sum()
        weights_at_open_before_rebalance = self._weights.copy()

        # Benchmarks begin at the first execution open.  From step two onward,
        # they remain fully invested through each overnight period.
        if step_index > 0:
            self._qqq_value *= 1.0 + self.overnight_returns[step_index, 0]
            self._spy_value *= 1.0 + self.overnight_returns[step_index, 1]

        turnover = 0.5 * float(
            np.abs(target_weights - self._weights).sum()
        )
        cost_fraction = self.transaction_cost_rate * turnover
        cost_value = self._portfolio_value * cost_fraction
        self._portfolio_value *= 1.0 - cost_fraction
        self._cumulative_turnover += turnover
        self._cumulative_transaction_cost += cost_value

        intraday_components = np.asarray(
            (
                *self.intraday_returns[step_index],
                self.cash_intraday_returns[step_index],
            ),
            dtype=np.float64,
        )
        intraday_portfolio_return = float(
            np.dot(target_weights, intraday_components)
        )
        intraday_growth = 1.0 + intraday_portfolio_return
        self._portfolio_value *= intraday_growth
        self._weights = (
            target_weights
            * (1.0 + intraday_components)
            / intraday_growth
        )
        self._weights /= self._weights.sum()

        self._qqq_value *= 1.0 + self.intraday_returns[step_index, 0]
        self._spy_value *= 1.0 + self.intraday_returns[step_index, 1]

        self._peak_value = max(self._peak_value, self._portfolio_value)
        drawdown = 1.0 - self._portfolio_value / self._peak_value
        self._max_drawdown = max(self._max_drawdown, drawdown)

        self._step_index += 1
        self._terminated = self._step_index == self.n_steps
        new_potential = self._score_components()["terminal_score"]
        reward = float(new_potential - self._potential)
        self._potential = new_potential

        return (
            self._observation(),
            reward,
            self._terminated,
            False,
            self._info(
                executed_step_index=step_index,
                feature_date=str(feature_date.astype("datetime64[D]")),
                target_date=str(target_date.astype("datetime64[D]")),
                action=int(action),
                target_weights=target_weights,
                weights_at_open_before_rebalance=(
                    weights_at_open_before_rebalance
                ),
                overnight_portfolio_return=overnight_portfolio_return,
                intraday_portfolio_return=intraday_portfolio_return,
                turnover=turnover,
                transaction_cost_fraction=cost_fraction,
                transaction_cost_value=cost_value,
            ),
        )

    def _score_components(self) -> dict[str, float | str]:
        portfolio_return = self._portfolio_value - 1.0
        spy_return = self._spy_value - 1.0
        qqq_return = self._qqq_value - 1.0
        (
            relative_score,
            reward_tier,
            alpha_strong,
            alpha_weak,
        ) = relative_performance_score(
            portfolio_return,
            spy_return,
            qqq_return,
            partial_credit=self.reward_config.partial_credit,
        )
        drawdown_component = (
            self.reward_config.drawdown_penalty * self._max_drawdown
        )
        turnover_component = (
            self.reward_config.turnover_regularizer
            * self._cumulative_turnover
        )
        terminal_score = (
            relative_score - drawdown_component - turnover_component
        )
        if self.reward_config.score_clip is not None:
            terminal_score = float(
                np.clip(
                    terminal_score,
                    -self.reward_config.score_clip,
                    self.reward_config.score_clip,
                )
            )
        return {
            "portfolio_return": float(portfolio_return),
            "spy_return": float(spy_return),
            "qqq_return": float(qqq_return),
            "alpha_strong": alpha_strong,
            "alpha_weak": alpha_weak,
            "relative_score": relative_score,
            "reward_tier": reward_tier,
            "max_drawdown": float(self._max_drawdown),
            "cumulative_turnover": float(self._cumulative_turnover),
            "cumulative_transaction_cost": float(
                self._cumulative_transaction_cost
            ),
            "drawdown_penalty": float(drawdown_component),
            "turnover_regularizer": float(turnover_component),
            "terminal_score": float(terminal_score),
        }

    def _observation(self) -> np.ndarray:
        score = self._score_components()
        progress = self._step_index / self.n_steps
        portfolio_fields = np.asarray(
            (
                *self._weights,
                score["portfolio_return"],
                score["spy_return"],
                score["qqq_return"],
                score["alpha_strong"],
                score["alpha_weak"],
                score["max_drawdown"],
                score["cumulative_turnover"],
                progress,
            ),
            dtype=np.float64,
        )
        feature_index = min(self._step_index, self.n_steps - 1)
        observation = np.concatenate(
            (self.features[feature_index], portfolio_fields)
        )
        return observation.astype(np.float32, copy=False)

    def _info(self, **step_details: Any) -> dict[str, Any]:
        observation_index = min(self._step_index, self.n_steps - 1)
        info: dict[str, Any] = {
            "observation_feature_date": str(
                self.feature_dates[observation_index].astype("datetime64[D]")
            ),
            "step_index": self._step_index,
            "weights": self._weights.copy(),
            "portfolio_value": float(self._portfolio_value),
            **self._score_components(),
        }
        info.update(step_details)
        return info
