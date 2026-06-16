#!/usr/bin/env python3
"""
Generate mock ERA5 monthly surface data for pipeline testing.

Creates NetCDF files that are format-compatible with the real CDS downloads,
so the entire pipeline (normalize → train → evaluate → plot) runs unchanged.

Simulated physics:
  - Seasonal cycle (sinusoidal, hemisphere-dependent)
  - Land-ocean contrast (simplified continent mask)
  - SST → t2m teleconnection (tropical Pacific SST anomaly → mid-latitude t2m
    response at 1-month lag) — this is the key predictability signal
  - Skewed precipitation (Gamma-like, many zeros)
  - Polar-only variables (snow depth, sea ice)
  - Small random noise

Usage:
  python -m data.mock_era5 --years 1995 2016 --output ./data/raw/era5_monthly
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

# ============================================================
# Grid
# ============================================================
N_LAT = 121   # 1.5°  spacing, -90 to 90
N_LON = 240   # 1.5°  spacing, 0 to 358.5

LATS = np.linspace(-90.0, 90.0, N_LAT, dtype=np.float32)
LONS = np.linspace(0.0, 358.5, N_LON, dtype=np.float32)

# Convert to radians for trig
LATS_RAD = np.deg2rad(LATS)
LONS_RAD = np.deg2rad(LONS)

# ============================================================
# Land-sea mask (simplified but realistic enough for testing)
# ============================================================
def make_land_mask():
    """Return land fraction (121, 240) — 1 = land, 0 = ocean."""
    lat2d = LATS[:, None]  # (121, 1)
    lon2d = LONS[None, :]  # (1, 240)

    # Simple continent shapes using sinusoidal patterns
    # Africa-Eurasia
    eurasia = (
        (lat2d > 20) & (lat2d < 75)
        & (lon2d > -10) & (lon2d < 180)
    ).astype(np.float32)

    # North America
    namerica = (
        (lat2d > 25) & (lat2d < 75)
        & (lon2d > -130) & (lon2d < -60)
    ).astype(np.float32)

    # South America
    samerica = (
        (lat2d > -55) & (lat2d < 10)
        & (lon2d > -80) & (lon2d < -35)
    ).astype(np.float32)

    # Africa
    africa = (
        (lat2d > -35) & (lat2d < 35)
        & (lon2d > -20) & (lon2d < 50)
    ).astype(np.float32)

    # Australia
    australia = (
        (lat2d > -40) & (lat2d < -10)
        & (lon2d > 110) & (lon2d < 155)
    ).astype(np.float32)

    # Antarctica
    antarctica = (lat2d < -65).astype(np.float32)

    land = np.clip(
        eurasia + namerica + samerica + africa + australia + antarctica, 0, 1
    )
    return land.astype(np.float32)


LAND = make_land_mask()
OCEAN = 1.0 - LAND

# ============================================================
# Base climatologies (per-variable background patterns)
# ============================================================
def seasonal_phase(month_idx: int) -> float:
    """Month index to seasonal phase (0 = Jan, peaks at Jul for NH)."""
    return 2.0 * np.pi * (month_idx - 6) / 12.0


def make_t2m(month_idx: int, rng: np.random.Generator) -> np.ndarray:
    """
    2m temperature (K).
    - Base: 288 K at equator, decreasing poleward
    - Seasonal amplitude: larger over land, larger at high latitudes
    """
    lat2d = LATS[:, None]
    phase = seasonal_phase(month_idx)

    # Equator-to-pole gradient
    base = 300.0 - 40.0 * np.abs(lat2d) / 90.0  # (121, 1)

    # Seasonal cycle: amplitude depends on latitude and land fraction
    seasonal_amp = (
        15.0 * np.abs(lat2d) / 90.0 * (LAND + 0.3 * OCEAN)
        + 5.0 * OCEAN
    )  # (121, 240)
    seasonal = seasonal_amp * np.cos(phase - np.pi * lat2d / 180.0)

    # SST teleconnection signal (see make_sst first, applied after)
    t2m_field = base + seasonal  # (121, 240)

    # Noise
    t2m_field += rng.normal(0, 1.0, (N_LAT, N_LON))

    return t2m_field.astype(np.float32)


def make_tp(month_idx: int, lat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Total precipitation (m). Skewed distribution — many zeros, occasional large values.
    """
    lat2d = lat[:, None]

    # Mean precipitation: highest in tropics, ITCZ
    mean_tp = 0.0001 + 0.0003 * np.exp(-0.5 * ((lat2d - 5.0) / 15.0) ** 2)

    # Seasonal shift of ITCZ
    shift = 5.0 * np.cos(seasonal_phase(month_idx))
    mean_shifted = 0.0001 + 0.0003 * np.exp(-0.5 * ((lat2d - shift) / 15.0) ** 2)

    # Gamma-like: exponential distribution clipped to non-negative
    tp_field = rng.exponential(mean_shifted) * OCEAN + rng.exponential(mean_tp * 1.2) * LAND
    tp_field = np.clip(tp_field, 0, 0.01)

    return tp_field.astype(np.float32)


def make_sst(month_idx: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sea surface temperature (K). Ocean only (NaN over land).
    Includes Niño 3.4-like variability for teleconnection signal.
    """
    lat2d = LATS[:, None]

    # Base SST: warm tropics, cold poles
    base = 302.0 - 30.0 * np.abs(lat2d) / 90.0

    # Seasonal cycle (muted compared to t2m)
    seasonal = 3.0 * np.cos(seasonal_phase(month_idx) - np.pi * lat2d / 180.0)

    # Start with lat-only profile, then broadcast to 2D grid
    sst_lat = base + seasonal  # (121, 1)

    # Niño-like SST anomaly in tropical Pacific (3-year ENSO cycle)
    enso_phase = 2.0 * np.pi * month_idx / 36.0
    enso_amp = 2.0 * np.cos(enso_phase + rng.normal(0, 0.3))
    enso_pattern = np.outer(
        np.exp(-0.5 * ((LATS - 0.0) / 8.0) ** 2),
        np.exp(-0.5 * ((LONS - 215.0) / 25.0) ** 2).astype(np.float32),
    )  # (121, 240)

    # Noise
    noise = rng.normal(0, 0.3, (N_LAT, N_LON))

    # Assemble: broadcast lat-only profile + 2D fields
    sst_field = sst_lat + enso_amp * enso_pattern + noise  # (121, 240)

    # NaN over land
    sst_field[LAND > 0.5] = np.nan

    return sst_field.astype(np.float32)


def get_nino34_sst_anomaly(
    sst_field: np.ndarray, sst_clim: np.ndarray, month_idx: int
) -> float:
    """Extract Niño 3.4 SST anomaly (used for teleconnection)."""
    ninio34_lon_mask = (LONS >= 190.0) & (LONS <= 240.0)
    ninio34_lat_mask = (LATS >= -5.0) & (LATS <= 5.0)
    region = sst_field[ninio34_lat_mask, :][:, ninio34_lon_mask]
    region_clim = sst_clim[ninio34_lat_mask, :][:, ninio34_lon_mask]
    anomaly = np.nanmean(region - region_clim)
    return float(anomaly)


def apply_teleconnection(
    t2m_field: np.ndarray,
    sst_anomaly: float,
    t2m_base: np.ndarray,
) -> np.ndarray:
    """
    Apply teleconnection: tropical Pacific SST anomaly → mid-latitude t2m response.

    Simulates the observed pattern where El Niño warming in the tropical Pacific
    leads to warmer temperatures over North America and cooler over the southern US.
    """
    lat2d = LATS[:, None]
    lon2d = LONS[None, :]

    # Response pattern: Pacific-North America (PNA-like)
    # Positive response over NW North America, negative over SE US
    pna_pattern = (
        np.exp(-0.5 * ((lat2d - 55.0) / 20.0) ** 2 - 0.5 * ((lon2d - 240.0) / 30.0) ** 2)
        - 0.5 * np.exp(-0.5 * ((lat2d - 35.0) / 15.0) ** 2 - 0.5 * ((lon2d - 270.0) / 25.0) ** 2)
    )

    # Scale: 1K SST anomaly → ~0.5K t2m response at 1-month lag
    response = 0.5 * sst_anomaly * pna_pattern
    return t2m_field + response.astype(np.float32)


def make_swvl1(month_idx: int, rng: np.random.Generator) -> np.ndarray:
    """Soil moisture (m³/m³), land only."""
    lat2d = LATS[:, None]
    # Wet in tropics, dry in deserts, frozen at high latitudes
    base = 0.25 - 0.15 * np.abs(lat2d) / 90.0
    seasonal_amp = 0.05
    seasonal = seasonal_amp * np.cos(seasonal_phase(month_idx) - np.pi * lat2d / 180.0)
    field = base + seasonal + rng.normal(0, 0.02, (N_LAT, N_LON))
    field = np.clip(field, 0.01, 0.5)
    field = np.where(LAND > 0.5, field, np.nan)
    return field.astype(np.float32)


def make_sd(month_idx: int, rng: np.random.Generator) -> np.ndarray:
    """Snow depth (m), high latitudes only, seasonal."""
    lat2d = LATS[:, None]
    # Only NH winter for simplicity
    lat_factor = np.clip((np.abs(lat2d) - 40.0) / 50.0, 0, 1)
    season_factor = 0.5 + 0.5 * np.cos(seasonal_phase(month_idx) + np.pi)  # winter peak

    field = 3.0 * lat_factor * season_factor + rng.normal(0, 0.1, (N_LAT, N_LON))
    field = np.clip(field, 0, 10.0)
    field = np.where(lat_factor > 0.1, field, np.nan)
    return field.astype(np.float32)


def make_siconc(month_idx: int, rng: np.random.Generator) -> np.ndarray:
    """Sea ice cover (0-1), polar oceans only."""
    lat2d = LATS[:, None]
    lat_factor = np.clip((np.abs(lat2d) - 60.0) / 30.0, 0, 1)
    season_factor = 0.5 + 0.5 * np.cos(seasonal_phase(month_idx) + np.pi)

    field = lat_factor * season_factor + rng.normal(0, 0.05, (N_LAT, N_LON))
    field = np.clip(field, 0, 1.0)
    field = np.where((OCEAN > 0.5) & (lat_factor > 0.05), field, np.nan)
    return field.astype(np.float32)


def make_msl(month_idx: int, rng: np.random.Generator) -> np.ndarray:
    """Mean sea level pressure (Pa)."""
    lat2d = LATS[:, None]
    base = 101325.0
    # Subtropical highs, polar lows
    pattern = (
        -2000.0 * np.exp(-0.5 * ((lat2d - 30.0) / 25.0) ** 2)
        + 1500.0 * np.exp(-0.5 * ((lat2d + 60.0) / 20.0) ** 2)
    )
    seasonal = 500.0 * np.cos(seasonal_phase(month_idx))
    field = base + pattern + seasonal + rng.normal(0, 200.0, (N_LAT, N_LON))
    return field.astype(np.float32)


def make_uv10(month_idx: int, rng: np.random.Generator) -> tuple:
    """10m wind components (m/s)."""
    lat2d = LATS[:, None]
    # Simple geostrophic-like pattern (pressure gradient drives wind)
    u = 5.0 * np.sin(lat2d * 3 * np.pi / 180.0) + rng.normal(0, 2.0, (N_LAT, N_LON))
    v = 3.0 * np.cos(lat2d * 2 * np.pi / 180.0) + rng.normal(0, 1.5, (N_LAT, N_LON))
    return u.astype(np.float32), v.astype(np.float32)


def make_tisr(month_idx: int) -> np.ndarray:
    """TOA incident solar radiation (W/m²)."""
    lat2d = LATS[:, None]
    phase = seasonal_phase(month_idx)
    # Solar constant ~ 1361, reduced by day length and zenith angle
    cos_lat = np.cos(lat2d)
    day_factor = 0.5 + 0.5 * np.cos(phase - lat2d * np.pi / 180.0)
    day_factor = np.clip(day_factor, 0, 1)
    field = 400.0 * cos_lat * day_factor  # (121, 1)
    field = np.broadcast_to(field, (N_LAT, N_LON)).copy()
    field = np.clip(field, 0, 500.0)
    return field.astype(np.float32)


def make_ssr(month_idx: int, rng: np.random.Generator) -> np.ndarray:
    """Surface net solar radiation (W/m²)."""
    tisr = make_tisr(month_idx)
    # Surface: reduced by albedo and absorption
    field = tisr * 0.55 + rng.normal(0, 10.0, (N_LAT, N_LON))
    return field.astype(np.float32)


def make_str(month_idx: int, rng: np.random.Generator) -> np.ndarray:
    """Surface net thermal radiation (W/m²) — negative (outgoing)."""
    lat2d = LATS[:, None]
    # More negative in warm regions, less negative in cold
    base = -150.0 + 60.0 * np.abs(lat2d) / 90.0
    seasonal = 20.0 * np.cos(seasonal_phase(month_idx) - np.pi * lat2d / 180.0)
    field = base + seasonal + rng.normal(0, 5.0, (N_LAT, N_LON))
    return field.astype(np.float32)


# ============================================================
# Variable factory
# ============================================================

VARIABLE_GENERATORS = {
    "t2m": make_t2m,
    "tp": make_tp,
    "sst": make_sst,
    "swvl1": make_swvl1,
    "sd": make_sd,
    "siconc": make_siconc,
    "msl": make_msl,
    "tisr": make_tisr,
    "ssr": make_ssr,
    "str": make_str,
}

# Variables that need special handling
VARIABLES_NEEDING_RNG = {"t2m", "tp", "sst", "swvl1", "sd", "siconc", "msl", "ssr", "str"}
VARIABLES_PURE = {"tisr"}  # deterministic
VARIABLES_TUPLE = {"u10", "v10"}  # generated as a pair


def generate_one_month(
    month_idx: int,
    prev_sst: np.ndarray,
    sst_clim_by_month: dict,
    rng: np.random.Generator,
) -> dict:
    """Generate all variables for one month."""
    data = {}

    # SST first (needed for teleconnection)
    sst = make_sst(month_idx, rng)
    data["sst"] = sst

    # Compute SST anomaly vs climatology for teleconnection
    clim_key = month_idx % 12
    if clim_key in sst_clim_by_month:
        nino_anom = get_nino34_sst_anomaly(sst, sst_clim_by_month[clim_key], month_idx)
    else:
        nino_anom = 0.0

    # t2m — with teleconnection from SST
    t2m_base = make_t2m(month_idx, rng)  # get base field without teleconnection
    t2m = apply_teleconnection(t2m_base, nino_anom, None)
    data["t2m"] = t2m

    # Precipitation
    data["tp"] = make_tp(month_idx, LATS, rng)

    # Land variables
    data["swvl1"] = make_swvl1(month_idx, rng)
    data["sd"] = make_sd(month_idx, rng)
    data["siconc"] = make_siconc(month_idx, rng)

    # Atmospheric
    data["msl"] = make_msl(month_idx, rng)
    u10, v10 = make_uv10(month_idx, rng)
    data["u10"] = u10
    data["v10"] = v10

    # Radiation
    data["tisr"] = make_tisr(month_idx)
    data["ssr"] = make_ssr(month_idx, rng)
    data["str"] = make_str(month_idx, rng)

    return data


# ============================================================
# Main
# ============================================================
def generate_data(years: tuple, output_dir: Path, seed: int = 42):
    """Generate mock ERA5 monthly data and write to NetCDF files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Build SST climatology first (needed for teleconnection)
    # We do two passes: first pass generates "climatology" months,
    # second pass generates actual data with teleconnection applied
    print("Building SST climatology...")
    sst_clim_by_month = {}
    for m in range(12):
        # Use a fixed seed for climatology
        clim_rng = np.random.default_rng(seed)
        sst_clim_by_month[m] = make_sst(m, clim_rng)

    # Generate data year by year
    for year in range(years[0], years[1] + 1):
        output_file = output_dir / f"era5_monthly_surface_{year}.nc"
        if output_file.exists():
            print(f"  [{year}] Already exists, skipping")
            continue

        print(f"  [{year}] Generating...")
        monthly_data = {var: [] for var in [
            "t2m", "tp", "sst", "swvl1", "sd", "siconc",
            "msl", "u10", "v10", "tisr", "ssr", "str",
        ]}

        for month in range(12):
            month_idx = (year - years[0]) * 12 + month
            data = generate_one_month(month_idx, None, sst_clim_by_month, rng)

            for var in monthly_data:
                monthly_data[var].append(data[var])

        # Stack months for each variable: (12, 121, 240)
        # Create xarray Dataset
        time_values = np.array([
            np.datetime64(f"{year}-{m+1:02d}-01") for m in range(12)
        ], dtype="datetime64[ns]")

        ds = xr.Dataset(
            data_vars={
                var: (["valid_time", "latitude", "longitude"],
                      np.stack(monthly_data[var]))
                for var in monthly_data
            },
            coords={
                "valid_time": time_values,
                "latitude": LATS,
                "longitude": LONS,
            },
        )

        # Write to NetCDF using scipy engine (more reliable on Windows)
        ds.to_netcdf(output_file, engine="scipy")
        ds.close()

        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  [{year}] Saved → {output_file.name} ({size_mb:.1f} MB)")

    print(f"\nDone. Data saved to {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate mock ERA5 monthly surface data"
    )
    parser.add_argument(
        "--years", type=int, nargs=2, default=[1995, 2016],
        help="Start and end year (inclusive), default: 1995 2016"
    )
    parser.add_argument(
        "--output", type=str, default="./data/raw/era5_monthly",
        help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    n_years = args.years[1] - args.years[0] + 1
    est_total = n_years * 30  # ~30 MB/year
    print(f"Mock ERA5 Monthly Surface Data Generator")
    print(f"  Grid: {N_LAT}×{N_LON} (1.5°)")
    print(f"  Variables: 12 surface")
    print(f"  Years: {args.years[0]}–{args.years[1]} ({n_years} years)")
    print(f"  Estimated total: ~{est_total} MB")
    print(f"  Output: {Path(args.output).resolve()}")
    print()

    generate_data((args.years[0], args.years[1]), Path(args.output), seed=args.seed)


if __name__ == "__main__":
    main()
