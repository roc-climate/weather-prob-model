"""
PyTorch Dataset for ERA5 monthly/weekly data.

Phase 1: Monthly data, with lead time = 1 month (~4 weeks).
Phase 2: Weekly data, with lead times 2-4 weeks.

Samples are constructed as:
  Input: x(t) — current state (all input variables)
  Target: y(t + lead_time) — t2m and tp at lead_time ahead

Data layout:
  Each .nc file contains [time, lat, lon] for multiple variables.
  After regridding to 1.5°, each timestep is ~ [n_vars, 121, 240].
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

import xarray as xr


# Mapping from ERA5 CDS variable names to our short names
ERA5_VAR_MAP = {
    "t2m": "t2m",
    "tp": "tp",
    "sst": "sst",
    "swvl1": "swvl1",
    "sd": "sd",
    "siconc": "siconc",
    "msl": "msl",
    "u10": "u10",
    "v10": "v10",
    "tisr": "tisr",
    "ssr": "ssr",
    "str": "str",
}

# ERA5 CDS internal variable names
CDS_VAR_NAMES = {
    "t2m": "t2m",
    "tp": "tp",
    "sst": "sst",
    "swvl1": "swvl1",
    "sd": "sd",
    "siconc": "siconc",
    "msl": "msl",
    "u10": "u10",
    "v10": "v10",
    "tisr": "tisr",
    "ssr": "ssr",
    "str": "str",
}


class WeatherDataset(Dataset):
    """
    Weather prediction dataset with configurable lead time.

    Args:
        data_dir: Path to directory containing .nc files
        variable_list: List of variable short names to load
        target_vars: Variables to predict (default: ["t2m", "tp"])
        lead_time: Number of time steps ahead to predict (1 = next month/week)
        years: Tuple of (start_year, end_year) inclusive
        normalize: Whether to normalize data (requires norm_stats)
        norm_stats: Dict with "mean" and "std" arrays for each variable
        time_dim: Name of time dimension in netCDF files
    """
    def __init__(
        self,
        data_dir: str,
        variable_list: list,
        target_vars: list = None,
        lead_time: int = 1,
        years: Tuple[int, int] = (1995, 2014),
        normalize: bool = True,
        norm_stats: Optional[Dict] = None,
        time_dim: str = "valid_time",
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.variable_list = variable_list
        self.target_vars = target_vars or ["t2m", "tp"]
        self.lead_time = lead_time
        self.normalize = normalize
        self.norm_stats = norm_stats

        # Load all data into memory (feasible for monthly data ~500 MB)
        self.data, self.time_index = self._load_data(years, time_dim)
        self.n_timesteps = self.data.shape[0]

        # Effective number of samples (must leave room for lead_time)
        self.n_samples = self.n_timesteps - lead_time

    def _load_data(
        self,
        years: Tuple[int, int],
        time_dim: str,
    ) -> Tuple[torch.Tensor, list]:
        """Load and stack all .nc files into a single tensor of shape (T, n_vars, H, W)."""
        start_year, end_year = years
        year_tensors = []
        all_times = []

        for year in range(start_year, end_year + 1):
            fname = f"era5_monthly_surface_{year}.nc"
            fpath = self.data_dir / fname

            if not fpath.exists():
                candidates = list(self.data_dir.glob(f"*{year}*.nc"))
                if candidates:
                    fpath = candidates[0]
                else:
                    print(f"Warning: {fname} not found, skipping year {year}")
                    continue

            ds = xr.open_dataset(fpath)

            # Collect per-variable arrays for this year: list of (T, H, W)
            var_arrays = []
            for var_name in self.variable_list:
                if var_name in ds:
                    arr = ds[var_name].values.astype(np.float32)
                    var_arrays.append(arr)
                else:
                    raise KeyError(f"Variable {var_name} not found in {fname}")

            ds.close()

            # Stack variables → (n_vars, T, H, W), then transpose → (T, n_vars, H, W)
            year_block = np.stack(var_arrays, axis=0)  # (n_vars, T, H, W)
            year_block = np.transpose(year_block, (1, 0, 2, 3))  # (T, n_vars, H, W)
            year_tensors.append(year_block)

            # Record times
            n_months = year_block.shape[0]
            for m in range(n_months):
                all_times.append(f"{year}-{m+1:02d}")

        if not year_tensors:
            raise FileNotFoundError(f"No data files found in {self.data_dir}")

        # Concatenate along time axis → (total_T, n_vars, H, W)
        all_arrays = np.concatenate(year_tensors, axis=0)

        print(f"  Loaded data shape: {all_arrays.shape} (T={all_arrays.shape[0]}, "
              f"vars={all_arrays.shape[1]}, H={all_arrays.shape[2]}, W={all_arrays.shape[3]})")

        return torch.from_numpy(all_arrays), all_times

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with:
              x_atmos:  (n_atmos_vars, H, W) atmospheric inputs
              x_slow:   (n_slow_vars, H, W) slow forcing inputs
              x_index:  (n_indices,) — placeholder (filled by collate or external)
              y_t2m:    (1, H, W) target t2m at lead_time
              y_tp:     (1, H, W) target tp at lead_time
        """
        input_step = idx
        target_step = idx + self.lead_time

        input_data = self.data[input_step]   # (n_vars, H, W)
        target_data = self.data[target_step]  # (n_vars, H, W)

        # Normalize if requested
        if self.normalize and self.norm_stats is not None:
            mean = self.norm_stats["mean"]  # (n_vars, 1, 1)
            std = self.norm_stats["std"]
            input_data = (input_data - mean) / (std + 1e-8)
            target_data = (target_data - mean) / (std + 1e-8)
            # Fill NaN (e.g., SST over land, soil moisture over ocean) with 0 = mean
            input_data = torch.nan_to_num(input_data, nan=0.0)
            target_data = torch.nan_to_num(target_data, nan=0.0)

        # Split into variable groups
        x_atmos = self._select_vars(input_data, ["msl", "u10", "v10", "tisr", "ssr", "str"])
        x_slow = self._select_vars(input_data, ["sst", "swvl1", "sd", "siconc"])
        y_t2m = self._select_vars(target_data, ["t2m"])
        y_tp = self._select_vars(target_data, ["tp"])

        return {
            "x_atmos": x_atmos,
            "x_slow": x_slow,
            "y_t2m": y_t2m,
            "y_tp": y_tp,
        }

    def _select_vars(self, data: torch.Tensor, var_names: list) -> torch.Tensor:
        """Select variables by name from the stacked tensor."""
        indices = [self.variable_list.index(v) for v in var_names if v in self.variable_list]
        if not indices:
            return torch.zeros(0, *data.shape[1:])  # Empty
        return data[indices]  # (n_selected, H, W)


class MonthlyToWeeklyAdapter(Dataset):
    """
    Wraps a monthly dataset to emulate weekly data for Phase 1 prototyping.

    Since Phase 1 uses monthly ERA5 data, this adapter treats each month
    as ~4 weeks and creates synthetic "weekly" samples by interpolation.
    This allows testing the full weekly pipeline before downloading daily data.

    In Phase 2, this is replaced by actual weekly-aggregated daily data.
    """
    def __init__(
        self,
        monthly_dataset: WeatherDataset,
        weeks_per_month: int = 4,
    ):
        self.monthly_dataset = monthly_dataset
        self.weeks_per_month = weeks_per_month

    def __len__(self) -> int:
        return self.monthly_dataset.n_samples * self.weeks_per_month

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        month_idx = idx // self.weeks_per_month
        week_offset = idx % self.weeks_per_month
        # Simplified: just return the monthly sample for all weeks
        return self.monthly_dataset[month_idx]
