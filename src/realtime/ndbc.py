"""NDBC (National Data Buoy Center) real-time buoy data fetcher.

Fetches latest observations from NOAA NDBC buoys.
Data source: https://www.ndbc.noaa.gov/data/realtime2/
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None

import pandas as pd

LOGGER = logging.getLogger(__name__)

NDBC_REALTIME_BASE = "https://www.ndbc.noaa.gov/data/realtime2/"


def fetch_ndbc_latest(
    station_id: str,
    data_type: str = "spec",
) -> pd.DataFrame:
    """Fetch latest observations from an NDBC buoy station.
    
    Args:
        station_id: NDBC station identifier (e.g., "46042", "51000")
        data_type: Type of data to fetch
            - "spec": Spectral wave data (wave height, period, direction)
            - "txt": Standard meteorological data
    
    Returns:
        DataFrame with timestamp index and observation data
    
    Note:
        Station IDs can be found at: https://www.ndbc.noaa.gov/
        Realtime2 data is typically the last 45 days.
    """
    if httpx is None:
        raise ImportError("httpx is required. Install with: pip install httpx")
    
    # Construct URL based on data type
    if data_type == "spec":
        url = f"{NDBC_REALTIME_BASE}{station_id}.spec"
    elif data_type == "txt":
        url = f"{NDBC_REALTIME_BASE}{station_id}.txt"
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()
            content = response.text
        
        # Parse the fixed-width or space-separated format
        lines = content.strip().split('\n')
        
        if len(lines) < 3:
            LOGGER.warning(f"Insufficient data from NDBC station {station_id}")
            return pd.DataFrame()
        
        # First line is header, second line is units
        header = lines[0].split()
        units = lines[1].split()
        data_lines = lines[2:]
        
        # Parse data rows
        rows = []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= len(header):
                rows.append(parts[:len(header)])
        
        if not rows:
            LOGGER.warning(f"No data rows found for NDBC station {station_id}")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=header)
        
        # Convert numeric columns
        for col in df.columns:
            if col not in ['#YY', 'YY', 'MM', 'DD', 'hh', 'mm']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        # Create timestamp from date/time columns
        if data_type == "spec":
            # Spectral data format: #YY MM DD hh mm
            if all(c in df.columns for c in ['#YY', 'MM', 'DD', 'hh', 'mm']):
                df['timestamp'] = pd.to_datetime(
                    df['#YY'] + '-' + df['MM'] + '-' + df['DD'] + ' ' + 
                    df['hh'] + ':' + df['mm'],
                    format='%Y-%m-%d %H:%M',
                    errors='coerce'
                )
        else:
            # Standard meteorological data format
            if all(c in df.columns for c in ['#YY', 'MM', 'DD', 'hh', 'mm']):
                df['timestamp'] = pd.to_datetime(
                    df['#YY'] + '-' + df['MM'] + '-' + df['DD'] + ' ' + 
                    df['hh'] + ':' + df['mm'],
                    format='%Y-%m-%d %H:%M',
                    errors='coerce'
                )
        
        if 'timestamp' in df.columns:
            df = df.dropna(subset=['timestamp'])
            df.set_index('timestamp', inplace=True)
            # Drop time component columns
            time_cols = ['#YY', 'YY', 'MM', 'DD', 'hh', 'mm']
            df = df.drop(columns=[c for c in time_cols if c in df.columns], errors='ignore')
        
        LOGGER.info(f"Fetched {len(df)} observations from NDBC station {station_id}")
        return df
        
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            LOGGER.warning(f"NDBC station {station_id} not found or no data available")
        else:
            LOGGER.error(f"HTTP error fetching NDBC data: {exc}")
        return pd.DataFrame()
    except Exception as exc:
        LOGGER.error(f"Failed to fetch NDBC data for station {station_id}: {exc}")
        return pd.DataFrame()


def prepare_ndbc_features(
    df: pd.DataFrame,
    data_type: str = "spec",
    target_features: List[str] = None,
) -> pd.DataFrame:
    """Convert NDBC data to model-compatible feature format.
    
    Args:
        df: DataFrame from fetch_ndbc_latest
        data_type: Type of source data ("spec" or "txt")
        target_features: Target feature names for model
    
    Returns:
        DataFrame with standardized feature columns
    """
    if target_features is None:
        target_features = ["Hs", "Hmax", "SST", "WindSpeed", "PeakWaveDirection"]
    
    result = pd.DataFrame(index=df.index)
    
    if data_type == "spec":
        # Spectral wave data columns (typical):
        # WVHT = Significant wave height (m)
        # SwH = Swell height (m)
        # WWH = Wind wave height (m)
        # MWD = Mean wave direction (deg)
        # APD = Average wave period (sec)
        
        if "WVHT" in df.columns and "Hs" in target_features:
            result["Hs"] = df["WVHT"]
        
        # Estimate Hmax from Hs
        if "Hs" in result.columns and "Hmax" in target_features:
            result["Hmax"] = result["Hs"] * 1.6
        
        if "MWD" in df.columns and "PeakWaveDirection" in target_features:
            result["PeakWaveDirection"] = df["MWD"]
    
    elif data_type == "txt":
        # Standard meteorological data columns (typical):
        # WVHT = Significant wave height (m)
        # DPD = Dominant wave period (sec)
        # MWD = Mean wave direction (deg)
        # WSPD = Wind speed (m/s)
        # WTMP = Water temperature (Celsius)
        
        if "WVHT" in df.columns and "Hs" in target_features:
            result["Hs"] = df["WVHT"]
        
        if "Hs" in result.columns and "Hmax" in target_features:
            result["Hmax"] = result["Hs"] * 1.6
        
        if "WSPD" in df.columns and "WindSpeed" in target_features:
            result["WindSpeed"] = df["WSPD"]
        
        if "WTMP" in df.columns and "SST" in target_features:
            result["SST"] = df["WTMP"]
        
        if "MWD" in df.columns and "PeakWaveDirection" in target_features:
            result["PeakWaveDirection"] = df["MWD"]
    
    return result


def find_nearest_station(
    latitude: float,
    longitude: float,
    max_distance_km: float = 500,
) -> Optional[str]:
    """Find nearest NDBC station to given coordinates.
    
    This is a simplified implementation. For production use, maintain
    a database of active stations with coordinates.
    
    Args:
        latitude: Target latitude
        longitude: Target longitude
        max_distance_km: Maximum search radius in kilometers
    
    Returns:
        Station ID if found within range, None otherwise
    """
    # Common Pacific and Atlantic stations for demonstration
    # In production, query the full station list from NDBC
    common_stations = {
        "46042": (36.785, -122.398),  # Monterey Bay, CA
        "46050": (44.642, -124.543),  # Stonewall Bank, OR
        "51000": (23.445, -162.279),  # NW Hawaii
        "41010": (28.878, -78.485),   # Canaveral East, FL
        "44025": (40.251, -73.164),   # Long Island, NY
    }
    
    from math import radians, sin, cos, asin, sqrt
    
    def haversine(lat1, lon1, lat2, lon2):
        """Calculate distance between two points on Earth in km."""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return 6371 * c  # Earth radius in km
    
    nearest_station = None
    min_distance = float('inf')
    
    for station_id, (st_lat, st_lon) in common_stations.items():
        distance = haversine(latitude, longitude, st_lat, st_lon)
        if distance < min_distance and distance <= max_distance_km:
            min_distance = distance
            nearest_station = station_id
    
    if nearest_station:
        LOGGER.info(f"Found nearest NDBC station {nearest_station} at {min_distance:.1f} km")
    else:
        LOGGER.warning(f"No NDBC station found within {max_distance_km} km of ({latitude}, {longitude})")
    
    return nearest_station
