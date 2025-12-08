"""Open-Meteo Marine API client for real-time ocean data.

Fetches point forecasts for wave height, direction, period, wind speed, and wind direction.
API docs: https://open-meteo.com/en/docs/marine-weather-api
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None

import pandas as pd

LOGGER = logging.getLogger(__name__)

OPEN_METEO_MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"


def fetch_open_meteo_marine(
    latitude: float,
    longitude: float,
    variables: Optional[List[str]] = None,
    hours: int = 48,
) -> pd.DataFrame:
    """Fetch marine forecast from Open-Meteo API.
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        variables: List of variables to fetch. Defaults to wave and wind parameters.
        hours: Number of forecast hours (default 48)
    
    Returns:
        DataFrame with time index and requested marine variables
    
    Example variables:
        - wave_height: Significant wave height in meters
        - wave_direction: Wave direction in degrees
        - wave_period: Wave period in seconds
        - wind_wave_height: Wind wave height in meters
        - swell_wave_height: Swell wave height in meters
        - wind_speed_10m: Wind speed at 10m in km/h
        - wind_direction_10m: Wind direction at 10m in degrees
    """
    if httpx is None:
        raise ImportError("httpx is required. Install with: pip install httpx")
    
    if variables is None:
        variables = [
            "wave_height",
            "wave_direction", 
            "wave_period",
            "wind_wave_height",
            "swell_wave_height",
            "wind_speed_10m",
            "wind_direction_10m",
        ]
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(variables),
        "forecast_hours": hours,
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(OPEN_METEO_MARINE_URL, params=params)
            response.raise_for_status()
            data = response.json()
        
        if "hourly" not in data:
            LOGGER.warning("No hourly data returned from Open-Meteo Marine API")
            return pd.DataFrame()
        
        hourly = data["hourly"]
        times = pd.to_datetime(hourly["time"])
        
        result = {"time": times}
        for var in variables:
            if var in hourly:
                result[var] = hourly[var]
            else:
                LOGGER.warning(f"Variable {var} not found in API response")
        
        df = pd.DataFrame(result)
        df.set_index("time", inplace=True)
        
        LOGGER.info(f"Fetched {len(df)} hourly marine forecasts for ({latitude}, {longitude})")
        return df
        
    except Exception as exc:
        LOGGER.error(f"Failed to fetch Open-Meteo marine data: {exc}")
        return pd.DataFrame()


def prepare_wave_features(
    df: pd.DataFrame,
    target_features: List[str] = None,
) -> pd.DataFrame:
    """Convert Open-Meteo marine data to model-compatible feature format.
    
    Args:
        df: DataFrame from fetch_open_meteo_marine
        target_features: List of target feature names expected by model
            Default: ["Hs", "Hmax", "SST", "WindSpeed", "PeakWaveDirection"]
    
    Returns:
        DataFrame with standardized column names
    """
    if target_features is None:
        target_features = ["Hs", "Hmax", "SST", "WindSpeed", "PeakWaveDirection"]
    
    # Map Open-Meteo variables to model features
    mapping = {
        "wave_height": "Hs",  # Significant wave height
        "wave_direction": "PeakWaveDirection",
        "wind_speed_10m": "WindSpeed",
    }
    
    result = pd.DataFrame(index=df.index)
    
    for open_meteo_var, model_feature in mapping.items():
        if open_meteo_var in df.columns and model_feature in target_features:
            result[model_feature] = df[open_meteo_var]
    
    # Estimate Hmax as 1.6 * Hs (common approximation)
    if "Hs" in result.columns and "Hmax" in target_features:
        result["Hmax"] = result["Hs"] * 1.6
    
    # Wind speed conversion from km/h to m/s if needed
    if "WindSpeed" in result.columns:
        result["WindSpeed"] = result["WindSpeed"] / 3.6
    
    # SST not available from marine API - would need separate data source
    # Fill with NaN for now
    if "SST" in target_features:
        result["SST"] = None
    
    return result
