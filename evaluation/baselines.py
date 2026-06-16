"""
Baseline models for comparison.

Phase 1 baselines:
  1. Climatology: Predict the climatological mean for each calendar month
  2. Persistence: Predict that the anomaly from the current month persists
  3. Damped persistence: Anomaly decays with e-folding time

These baselines establish the minimum bar the model must beat.
"""

import numpy as np
from typing import Dict


class ClimatologyBaseline:
    """
    Predict the climatological mean for each calendar month.

    This is the simplest possible forecast: "next month will look like
    the average of all past instances of that calendar month."

    For temperature this is surprisingly competitive at S2S timescales.
    """
    def __init__(self):
        self.climatology = {}  # {month: (H, W) mean}

    def fit(self, data: np.ndarray, times: list):
        """
        Compute monthly climatology from training data.

        Args:
            data: (T, H, W) variable data over training period
            times: list of datetime objects (length T)
        """
        from collections import defaultdict
        monthly_data = defaultdict(list)

        for i, t in enumerate(times):
            month = t.month if hasattr(t, 'month') else (i % 12 + 1)
            monthly_data[month].append(data[i])

        for month, values in monthly_data.items():
            self.climatology[month] = np.mean(values, axis=0)

    def predict(self, month: int) -> np.ndarray:
        """Return (H, W) climatological prediction for the given month."""
        return self.climatology.get(month, np.zeros(1))


class PersistenceBaseline:
    """
    Predict that the current anomaly persists.

    y_pred(t + Δt) = climatology(target_month) + anomaly(t)

    This works well for SST (long memory) but poorly for atmospheric variables
    at 3+ week lead times.
    """
    def __init__(self, climatology: ClimatologyBaseline):
        self.climatology = climatology

    def predict(
        self,
        current: np.ndarray,
        current_month: int,
        target_month: int,
    ) -> np.ndarray:
        """
        Args:
            current: (H, W) current state
            current_month: integer month of current state
            target_month: integer month to predict

        Returns:
            (H, W) persistence prediction
        """
        current_clim = self.climatology.predict(current_month)
        target_clim = self.climatology.predict(target_month)
        anomaly = current - current_clim
        return target_clim + anomaly


class DampedPersistenceBaseline:
    """
    Predict that the anomaly decays exponentially.

    y_pred(t + Δt) = climatology(target) + anomaly(t) * exp(-Δt / τ)

    where τ is the e-folding time scale (in time steps).

    Typical τ values:
      - Atmospheric variables (t2m, msl): 5-10 days (τ ≈ 1 week)
      - SST: 2-6 months (τ ≈ 8-24 weeks)
      - Soil moisture: 2-8 weeks
    """
    def __init__(self, climatology: ClimatologyBaseline, e_folding_time: float = 2.0):
        """
        Args:
            climatology: ClimatologyBaseline instance
            e_folding_time: τ in time steps (default: 2 for ~2 weeks for atm)
        """
        self.climatology = climatology
        self.e_folding_time = e_folding_time

    def predict(
        self,
        current: np.ndarray,
        current_month: int,
        target_month: int,
        n_steps: int = 1,
    ) -> np.ndarray:
        """
        Args:
            current: (H, W) current state
            current_month: integer month
            target_month: integer month
            n_steps: number of time steps ahead (ΔT)

        Returns:
            (H, W) damped persistence prediction
        """
        current_clim = self.climatology.predict(current_month)
        target_clim = self.climatology.predict(target_month)
        anomaly = current - current_clim
        decay = np.exp(-n_steps / self.e_folding_time)
        return target_clim + anomaly * decay


def evaluate_baseline(
    baseline,
    data: np.ndarray,
    times: list,
    lead_time: int,
) -> Dict[str, float]:
    """
    Evaluate a baseline over the dataset.

    Args:
        baseline: One of the baseline objects
        data: (T, H, W) variable data
        times: list of datetime objects
        lead_time: forecast lead time in time steps

    Returns:
        dict with rmse, mae, and (if applicable) crps metrics
    """
    n = len(data) - lead_time
    errors = []

    for i in range(n):
        current = data[i]
        target = data[i + lead_time]

        current_month = times[i].month if hasattr(times[i], 'month') else (i % 12 + 1)
        target_month = times[i + lead_time].month if hasattr(times[i + lead_time], 'month') else ((i + lead_time) % 12 + 1)

        if isinstance(baseline, ClimatologyBaseline):
            pred = baseline.predict(target_month)
        elif isinstance(baseline, PersistenceBaseline):
            pred = baseline.predict(current, current_month, target_month)
        elif isinstance(baseline, DampedPersistenceBaseline):
            pred = baseline.predict(current, current_month, target_month, lead_time)
        else:
            raise ValueError(f"Unknown baseline type: {type(baseline)}")

        error = (pred - target) ** 2
        errors.append(error)

    errors = np.array(errors)
    rmse = float(np.sqrt(np.mean(errors)))
    mae = float(np.mean(np.abs(errors)))

    return {"rmse": rmse, "mae": mae}
