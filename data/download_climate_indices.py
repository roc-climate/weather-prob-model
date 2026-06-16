#!/usr/bin/env python3
"""
Download climate indices for S2S prediction.

Sources (all public, free):
  - MJO (RMM1, RMM2):  BOM Australia — http://www.bom.gov.au/climate/mjo/
  - Nino 3.4:           NOAA CPC — https://www.cpc.ncep.noaa.gov/
  - NAO, AO:            NOAA CPC — https://www.cpc.ncep.noaa.gov/

The indices are small text/CSV files (KB level).

Usage:
  python download_climate_indices.py --output ./data/raw/climate_indices
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import numpy as np


# ---- MJO RMM indices (BOM Australia) ----
# BOM provides a text file with daily RMM values
# Format: year month day RMM1 RMM2 phase amplitude (missing=-999)
MJO_URL = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"


def download_mjo(output_dir: Path) -> pd.DataFrame:
    """Download MJO RMM indices from BOM and return as DataFrame."""
    print("Downloading MJO RMM indices from BOM Australia...")
    output_file = output_dir / "mjo_rmm_daily.csv"

    with urlopen(MJO_URL) as resp:
        lines = resp.read().decode("utf-8").splitlines()

    # Skip the first 2 header lines
    data = []
    for line in lines[2:]:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            rmm1 = float(parts[3])
            rmm2 = float(parts[4])
            phase = float(parts[5])
            amplitude = float(parts[6])
            # MJO missing data uses 999 or -999
            if abs(rmm1) > 900 or abs(rmm2) > 900:
                rmm1 = np.nan
                rmm2 = np.nan
                amplitude = np.nan
            data.append({
                "date": datetime(year, month, day),
                "rmm1": rmm1,
                "rmm2": rmm2,
                "phase": phase,
                "amplitude": amplitude,
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"  Saved {len(df)} rows to {output_file}")
    return df


# ---- Nino 3.4 (NOAA CPC) ----
# NOAA provides monthly Nino 3.4 SST anomaly
NINO34_URL = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.81-10.ascii"


def download_nino34(output_dir: Path) -> pd.DataFrame:
    """Download Nino 3.4 monthly anomaly from NOAA CPC."""
    print("Downloading Nino 3.4 index from NOAA CPC...")
    output_file = output_dir / "nino34_monthly.csv"

    with urlopen(NINO34_URL) as resp:
        lines = resp.read().decode("utf-8").splitlines()

    data = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        try:
            year = int(parts[0])
            month = int(parts[1])
            anomaly = float(parts[2])
            data.append({
                "date": datetime(year, month, 15),
                "nino34": anomaly,
                "nino34_anom_3m": np.nan,  # will fill from NOAA's 3-month running mean
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(data)

    # Compute 3-month running mean (standard ONI-style)
    if len(df) >= 3:
        df["nino34_3m"] = df["nino34"].rolling(3, center=True).mean()

    df.to_csv(output_file, index=False)
    print(f"  Saved {len(df)} rows to {output_file}")
    return df


# ---- NAO and AO (NOAA CPC) ----
NAO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table"
AO_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table"


def download_nao_ao(output_dir: Path) -> pd.DataFrame:
    """Download NAO and AO monthly indices from NOAA CPC."""
    print("Downloading NAO/AO indices from NOAA CPC...")
    output_file = output_dir / "nao_ao_monthly.csv"

    all_records = {}

    # NAO
    try:
        with urlopen(NAO_URL) as resp:
            lines = resp.read().decode("utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                year_val = int(parts[0])
                month_val = int(parts[1])
                val = float(parts[2])
                if abs(val) > 900:  # missing
                    val = np.nan
                key = (year_val, month_val)
                if key not in all_records:
                    all_records[key] = {}
                all_records[key]["nao"] = val
            except (ValueError, IndexError):
                continue
    except Exception as e:
        print(f"  Warning: Could not download NAO: {e}")

    # AO
    try:
        with urlopen(AO_URL) as resp:
            lines = resp.read().decode("utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                year_val = int(parts[0])
                month_val = int(parts[1])
                val = float(parts[2])
                if abs(val) > 900:
                    val = np.nan
                key = (year_val, month_val)
                if key not in all_records:
                    all_records[key] = {}
                all_records[key]["ao"] = val
            except (ValueError, IndexError):
                continue
    except Exception as e:
        print(f"  Warning: Could not download AO: {e}")

    # Convert to DataFrame
    records = []
    for (year_val, month_val), vals in sorted(all_records.items()):
        records.append({
            "date": datetime(year_val, month_val, 15),
            "nao": vals.get("nao", np.nan),
            "ao": vals.get("ao", np.nan),
        })

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"  Saved {len(df)} rows to {output_file}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download climate indices for S2S prediction"
    )
    parser.add_argument(
        "--output", type=str, default="./data/raw/climate_indices",
        help="Output directory, default: ./data/raw/climate_indices"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Climate Indices Download")
    print(f"  Output: {output_dir.resolve()}")
    print()

    try:
        df_mjo = download_mjo(output_dir)
    except Exception as e:
        print(f"  Error downloading MJO: {e}")
        df_mjo = None

    try:
        df_nino34 = download_nino34(output_dir)
    except Exception as e:
        print(f"  Error downloading Nino34: {e}")
        df_nino34 = None

    try:
        df_nao_ao = download_nao_ao(output_dir)
    except Exception as e:
        print(f"  Error downloading NAO/AO: {e}")
        df_nao_ao = None

    print()
    print("Climate indices download complete.")
    print()
    print("Note: MJO is daily, Nino34/NAO/AO are monthly.")
    print("In Phase 1 (monthly data), use the monthly average of MJO amplitude/phase.")
    print("In Phase 2 (weekly data), use weekly averages of daily MJO.")


if __name__ == "__main__":
    main()
