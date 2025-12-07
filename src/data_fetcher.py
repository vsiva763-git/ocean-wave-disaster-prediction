"""Data fetching and preprocessing for multimodal ocean wave datasets.

Designed for Google Colab-friendly workflows with graceful fallbacks when
remote APIs are unavailable. Provides:
- Satellite composites (optical, SAR, SST) via Google Earth Engine when available.
- Reanalysis time-series (ERA5) and optional NDBC buoy data.
- Synthetic data generation fallback to keep pipelines runnable offline.
- Sliding-window sequence builder aligned to image timestamps.

Authentication notes (also see README):
- Earth Engine: `earthengine authenticate` then `ee.Authenticate()` in Python.
- Copernicus CDS/ERA5: place your CDS API key in `~/.cdsapirc` or use `cdsapi.Client(url,key)`.
- Sentinel Hub: set `SH_CLIENT_ID` and `SH_CLIENT_SECRET` env vars or config file.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from joblib import dump
from sklearn.preprocessing import StandardScaler

# Optional heavy deps
try:
    import ee  # type: ignore
except Exception:  # pragma: no cover - optional
    ee = None

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover - optional
    xr = None

try:
    import cdsapi  # type: ignore
except Exception:  # pragma: no cover - optional
    cdsapi = None

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional
    requests = None


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)

DEFAULT_FEATURES = [
    "Hs",  # significant wave height
    "Hmax",  # max wave height or peak period proxy
    "SST",  # sea surface temperature
    "WindSpeed",
    "PeakWaveDirection",
]

# Common bounding boxes for waters around India (min_lon, min_lat, max_lon, max_lat)
PRESET_BBOXES: Dict[str, Tuple[float, float, float, float]] = {
    "india_all": (66.0, 5.0, 100.0, 23.0),  # Arabian Sea + Bay of Bengal
    "bay_of_bengal": (80.0, 5.0, 101.0, 22.0),
    "arabian_sea": (60.0, 5.0, 78.0, 23.0),
    "andaman": (90.0, 6.0, 100.0, 15.0),
    "lakshadweep": (70.0, 8.0, 76.0, 14.0),
}


@dataclass
class FetchResult:
    csv_path: str
    image_dir: str
    summary_path: str


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def bbox_to_geojson(bbox: Tuple[float, float, float, float]) -> Dict:
    min_lon, min_lat, max_lon, max_lat = bbox
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [min_lon, min_lat],
                            [min_lon, max_lat],
                            [max_lon, max_lat],
                            [max_lon, min_lat],
                            [min_lon, min_lat],
                        ]
                    ],
                },
                "properties": {},
            }
        ],
    }


def save_png(array: np.ndarray, path: str) -> None:
    array = np.clip(array, 0, 1)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    img = Image.fromarray((array * 255).astype(np.uint8))
    img.save(path)


# ---------------------------------------------------------------------------
# Earth Engine helpers (best-effort; may need user auth)
# ---------------------------------------------------------------------------

def try_init_ee() -> bool:
    if ee is None:
        LOGGER.warning("earthengine-api not installed. Install with `pip install earthengine-api`." )
        return False
    try:
        ee.Initialize()
        return True
    except Exception as exc:  # pragma: no cover - user auth issue
        LOGGER.warning("Failed to initialize Earth Engine: %s", exc)
        LOGGER.info("Run: !pip install earthengine-api && earthengine authenticate")
        return False


def fetch_optical_composite_ee(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    img_size: int,
    out_png: str,
) -> Optional[str]:
    if not try_init_ee():
        return None
    try:
        geometry = ee.Geometry.Rectangle(list(bbox))
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        )
        def _mask_clouds(img):
            qa = img.select("QA60")
            cloud_bit = 1 << 10
            cirrus_bit = 1 << 11
            mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
            return img.updateMask(mask)
        collection = collection.map(_mask_clouds)
        composite = collection.median().clip(geometry)
        rgb = composite.select(["B4", "B3", "B2"]).divide(10000)
        url = rgb.getThumbURL({"region": geometry, "dimensions": img_size})
        if requests is None:
            LOGGER.warning("requests not available to download thumbnail from EE; skipping")
            return None
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(out_png, "wb") as f:
            f.write(resp.content)
        LOGGER.info("Saved optical composite to %s", out_png)
        return out_png
    except Exception as exc:  # pragma: no cover - remote failures
        LOGGER.warning("EE optical composite failed: %s", exc)
        return None


def fetch_sar_composite_ee(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    img_size: int,
    out_png: str,
) -> Optional[str]:
    if not try_init_ee():
        return None
    try:
        geometry = ee.Geometry.Rectangle(list(bbox))
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        )
        composite = collection.median().select(["VV", "VH"]).clip(geometry)
        vv = composite.select("VV").unitScale(-25, 5)
        vh = composite.select("VH").unitScale(-30, 5)
        rgb = composite.addBands(vv).addBands(vh)
        url = rgb.getThumbURL({"region": geometry, "dimensions": img_size})
        if requests is None:
            LOGGER.warning("requests not available to download thumbnail from EE; skipping")
            return None
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(out_png, "wb") as f:
            f.write(resp.content)
        LOGGER.info("Saved SAR composite to %s", out_png)
        return out_png
    except Exception as exc:  # pragma: no cover - remote failures
        LOGGER.warning("EE SAR composite failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Time-series fetching (ERA5 / NDBC) with fallbacks
# ---------------------------------------------------------------------------

def fetch_era5_timeseries(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    variables: Sequence[str],
    interval_hours: int = 3,
) -> pd.DataFrame:
    """Best-effort ERA5 pull. Falls back to synthetic data if cdsapi missing.

    Returns DataFrame with datetime index and requested variable columns.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = (min_lon + max_lon) / 2
    lat = (min_lat + max_lat) / 2
    dates = pd.date_range(start=start_date, end=end_date, freq=f"{interval_hours}H")

    if cdsapi is None:
        LOGGER.warning("cdsapi not installed; generating synthetic reanalysis data")
        return _synthetic_timeseries(dates, variables)

    try:
        client = cdsapi.Client()  # expects ~/.cdsapirc
        LOGGER.info("Requesting ERA5 data (this may take a while)...")
        # Minimal example request; users should adjust for production throughput.
        request = {
            "product_type": "reanalysis",
            "variable": list(variables),
            "year": sorted({d.strftime("%Y") for d in dates}),
            "month": sorted({d.strftime("%m") for d in dates}),
            "day": sorted({d.strftime("%d") for d in dates}),
            "time": [f"{h:02d}:00" for h in range(0, 24, interval_hours)],
            "area": [max_lat, min_lon, min_lat, max_lon],
            "format": "netcdf",
        }
        target = "era5_subset.nc"
        client.retrieve("reanalysis-era5-single-levels", request, target)
        if xr is None:
            LOGGER.warning("xarray not available; using synthetic fallback instead")
            return _synthetic_timeseries(dates, variables)
        ds = xr.open_dataset(target)
        data = {var: ds[var].mean(dim=[dim for dim in ds[var].dims if dim not in ["time"]]).values for var in variables if var in ds}
        df = pd.DataFrame(data, index=pd.to_datetime(ds["time"].values))
        ds.close()
        os.remove(target)
        return df
    except Exception as exc:  # pragma: no cover - network/API issues
        LOGGER.warning("ERA5 fetch failed (%s); using synthetic data", exc)
        return _synthetic_timeseries(dates, variables)


def fetch_ndbc_buoy_timeseries(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """Optional NDBC buoy pull (best-effort)."""
    if requests is None:
        return None
    # This is a simplified placeholder; real buoy lookup would query NDBC station list by distance.
    try:
        LOGGER.info("Attempting NDBC buoy fetch (placeholder endpoint)...")
        # Placeholder URL; user should replace with actual station CSV if known
        return None
    except Exception:  # pragma: no cover
        return None


def _synthetic_timeseries(dates: pd.DatetimeIndex, variables: Sequence[str]) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {}
    t = np.linspace(0, 2 * np.pi, len(dates))
    for i, var in enumerate(variables):
        base = rng.normal(loc=0.0, scale=1.0, size=len(dates)) + 0.2 * np.sin(t + i)
        data[var] = base
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def normalize_image(arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    stats = {"min": float(arr.min()), "max": float(arr.max())}
    if stats["max"] - stats["min"] > 1e-6:
        arr = (arr - stats["min"]) / (stats["max"] - stats["min"])
    arr = np.clip(arr, 0, 1)
    return arr, stats


def build_sequences(
    df: pd.DataFrame,
    seq_len: int,
    feature_order: Sequence[str],
    align_time: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Build sliding windows ending at align_time (or the end of df)."""
    if align_time is not None:
        df = df[df.index <= align_time]
    df = df.sort_index()
    if len(df) < seq_len:
        df = df.reindex(pd.date_range(df.index.min(), periods=seq_len, freq=(df.index[1]-df.index[0]) if len(df)>1 else "1H"))
    df = df.interpolate().fillna(method="ffill").fillna(method="bfill")

    scaler = StandardScaler()
    values = df[feature_order].to_numpy()
    values = scaler.fit_transform(values)

    windows = []
    timestamps = []
    for i in range(len(values) - seq_len + 1):
        windows.append(values[i : i + seq_len])
        timestamps.append(df.index[i + seq_len - 1])
    if not windows:
        windows = [values[-seq_len:]]
        timestamps = [df.index[-1]]

    flat_cols = [f"t{t}_f{f}" for t in range(seq_len) for f in range(len(feature_order))]
    rows = []
    for w, ts in zip(windows, timestamps):
        row = {col: float(val) for col, val in zip(flat_cols, w.flatten())}
        row["timestamp"] = ts.isoformat()
        rows.append(row)
    return pd.DataFrame(rows), scaler


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def fetch_region_dataset(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    out_dir: str,
    img_size: int = 128,
    seq_len: int = 12,
    seq_features: int = 5,
) -> FetchResult:
    """Fetch imagery + time-series and build CSV suitable for training.

    Returns paths to CSV, image directory, and summary JSON.
    """
    ensure_dir(out_dir)
    image_dir = ensure_dir(os.path.join(out_dir, "images"))
    summary = {
        "bbox": bbox,
        "start_date": start_date,
        "end_date": end_date,
        "img_size": img_size,
        "seq_len": seq_len,
        "seq_features": seq_features,
        "notes": [],
    }

    optical_path = os.path.join(image_dir, "optical.png")
    sar_path = os.path.join(image_dir, "sar.png")
    optical = fetch_optical_composite_ee(bbox, start_date, end_date, img_size, optical_path)
    sar = fetch_sar_composite_ee(bbox, start_date, end_date, img_size, sar_path)

    if optical is None:
        summary["notes"].append("Optical composite unavailable; using synthetic image")
        optical_array = np.clip(np.random.rand(img_size, img_size, 3), 0, 1)
        save_png(optical_array, optical_path)
        optical = optical_path
    if sar is None:
        summary["notes"].append("SAR composite unavailable; using synthetic image")
        sar_array = np.clip(np.random.rand(img_size, img_size, 3), 0, 1)
        save_png(sar_array, sar_path)
        sar = sar_path

    # Simple fusion by stacking optical RGB with SAR VV/VH mean (as extra channels if desired)
    fused = Image.open(optical).convert("RGB")
    fused_arr = np.array(fused).astype(np.float32) / 255.0
    summary["image_used"] = optical

    # Time-series
    variables = [
        "wind_speed",  # maps to WindSpeed
        "sst",         # SST
        "msl",         # mean sea-level pressure proxy for storms
        "swh",         # significant wave height
        "pp1d",        # peak period or peak wave direction proxy
    ][:seq_features]
    ts_df = fetch_era5_timeseries(bbox, start_date, end_date, variables)
    buoy_df = fetch_ndbc_buoy_timeseries(bbox, start_date, end_date)
    if buoy_df is not None:
        summary["notes"].append("Merged NDBC buoy observations where available")
        ts_df = ts_df.combine_first(buoy_df)

    seq_df, scaler = build_sequences(ts_df, seq_len, variables)

    # Save scaler
    scaler_path = os.path.join(out_dir, "sequence_scaler.pkl")
    dump(scaler, scaler_path)

    # Build CSV
    flat_cols = [f"t{t}_f{f}" for t in range(seq_len) for f in range(seq_features)]
    records = []
    for _, row in seq_df.iterrows():
        rec = {"image": os.path.relpath(optical, out_dir), "label": -1}
        for c in flat_cols:
            rec[c] = row.get(c, 0.0)
        rec["timestamp"] = row["timestamp"]
        records.append(rec)

    csv_path = os.path.join(out_dir, "dataset.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)

    summary_path = os.path.join(out_dir, "summary.json")
    summary.update({"records": len(records), "scaler_path": scaler_path})
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("Dataset ready: %s (images at %s)", csv_path, image_dir)
    return FetchResult(csv_path=csv_path, image_dir=image_dir, summary_path=summary_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch multimodal data for ocean wave prediction")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=False,
        help="min_lon min_lat max_lon max_lat (required unless --preset is used)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(PRESET_BBOXES.keys()),
        help="Preset region covering Indian waters (overrides --bbox)",
    )
    parser.add_argument("--start_date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--seq_features", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.preset:
        bbox = PRESET_BBOXES[args.preset]
    elif args.bbox:
        bbox = tuple(args.bbox)  # type: ignore
    else:
        LOGGER.error("You must provide either --preset or --bbox")
        sys.exit(1)
    fetch_region_dataset(
        bbox=bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        out_dir=args.out_dir,
        img_size=args.img_size,
        seq_len=args.seq_len,
        seq_features=args.seq_features,
    )


if __name__ == "__main__":
    main()
