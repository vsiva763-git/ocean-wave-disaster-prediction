"""Enhanced Kanyakumari Real-Time Data Fetcher.

Comprehensive data aggregator for ocean monitoring focused on Kanyakumari region.
Fetches data from multiple free APIs:
- Open-Meteo Marine API (wave forecasts and conditions)
- Open-Meteo Weather API (atmospheric conditions)
- USGS Earthquake Catalog (seismic events)
- NOAA Tsunami Warning Centers (official bulletins)

Features:
- Tsunami risk assessment based on earthquake parameters
- Wave condition analysis and forecasting
- Climatic condition monitoring
- Historical tsunami data for the region
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from math import radians, sin, cos, asin, sqrt
from enum import Enum

import numpy as np
import pandas as pd

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    feedparser = None

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


# ============================================================================
# KANYAKUMARI LOCATION CONFIGURATION
# ============================================================================

# Kanyakumari - Southernmost tip of India
KANYAKUMARI_LAT = 8.0883
KANYAKUMARI_LON = 77.5385

# Extended monitoring region (Arabian Sea + Indian Ocean + Bay of Bengal)
KANYAKUMARI_REGION = {
    "name": "Kanyakumari Coastal Region",
    "center": {"lat": KANYAKUMARI_LAT, "lon": KANYAKUMARI_LON},
    "bounds": {
        "min_lat": 5.0,   # South into Indian Ocean
        "max_lat": 12.0,  # Northern Kerala/Tamil Nadu
        "min_lon": 74.0,  # Arabian Sea
        "max_lon": 82.0,  # Bay of Bengal
    },
    "timezone": "Asia/Kolkata",
    "country": "India",
    "state": "Tamil Nadu"
}

# Nearby ocean monitoring points
MONITORING_POINTS = {
    "kanyakumari_main": {"lat": 8.0883, "lon": 77.5385, "name": "Kanyakumari"},
    "kovalam": {"lat": 8.3684, "lon": 77.0214, "name": "Kovalam"},
    "tuticorin": {"lat": 8.7642, "lon": 78.1348, "name": "Tuticorin"},
    "rameshwaram": {"lat": 9.2876, "lon": 79.3129, "name": "Rameshwaram"},
    "trivandrum": {"lat": 8.5074, "lon": 76.9558, "name": "Thiruvananthapuram"},
}

# Historical tsunamis affecting Kanyakumari
HISTORICAL_TSUNAMIS = [
    {
        "date": "2004-12-26",
        "name": "Indian Ocean Tsunami (Boxing Day)",
        "earthquake_magnitude": 9.1,
        "epicenter": {"lat": 3.316, "lon": 95.854},
        "location": "Sumatra, Indonesia",
        "max_wave_height_m": 10.0,
        "deaths_kanyakumari": 802,
        "deaths_total": 230000,
        "arrival_time_minutes": 120,
        "warning_issued": False,
        "description": "Most devastating tsunami in recorded history"
    },
    {
        "date": "1883-08-27",
        "name": "Krakatoa Volcanic Tsunami",
        "earthquake_magnitude": 6.0,
        "epicenter": {"lat": -6.102, "lon": 105.423},
        "location": "Krakatoa, Indonesia",
        "max_wave_height_m": 1.5,
        "arrival_time_minutes": 720,
        "description": "Volcanic eruption-induced tsunami"
    },
    {
        "date": "1941-06-26",
        "name": "Andaman Sea Earthquake Tsunami",
        "earthquake_magnitude": 7.7,
        "epicenter": {"lat": 12.5, "lon": 92.5},
        "location": "Andaman Islands",
        "max_wave_height_m": 1.0,
        "description": "Moderate tsunami affecting eastern coast"
    }
]


# ============================================================================
# API ENDPOINTS
# ============================================================================

class APIEndpoints:
    """API endpoint URLs for data fetching."""
    OPEN_METEO_MARINE = "https://marine-api.open-meteo.com/v1/marine"
    OPEN_METEO_WEATHER = "https://api.open-meteo.com/v1/forecast"
    USGS_EARTHQUAKE = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    PTWC_RSS = "https://www.tsunami.gov/events/xml/PHEBxml.xml"  # Pacific
    NTWC_RSS = "https://www.tsunami.gov/events/xml/WCEBxml.xml"  # West Coast/Alaska


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class RiskLevel(Enum):
    """Risk level enumeration."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class WaveCondition(Enum):
    """Wave condition classification."""
    CALM = "calm"           # < 0.5m
    SLIGHT = "slight"       # 0.5 - 1.25m
    MODERATE = "moderate"   # 1.25 - 2.5m
    ROUGH = "rough"         # 2.5 - 4m
    VERY_ROUGH = "very_rough"  # 4 - 6m
    HIGH = "high"           # 6 - 9m
    VERY_HIGH = "very_high" # 9 - 14m
    PHENOMENAL = "phenomenal"  # > 14m


@dataclass
class OceanConditions:
    """Current ocean conditions data class."""
    wave_height_m: Optional[float] = None
    wave_period_s: Optional[float] = None
    wave_direction_deg: Optional[float] = None
    swell_height_m: Optional[float] = None
    swell_period_s: Optional[float] = None
    swell_direction_deg: Optional[float] = None
    wind_wave_height_m: Optional[float] = None
    sea_surface_temp_c: Optional[float] = None
    timestamp: Optional[str] = None
    
    @property
    def wave_condition(self) -> WaveCondition:
        """Classify wave conditions."""
        if self.wave_height_m is None:
            return WaveCondition.CALM
        h = self.wave_height_m
        if h < 0.5:
            return WaveCondition.CALM
        elif h < 1.25:
            return WaveCondition.SLIGHT
        elif h < 2.5:
            return WaveCondition.MODERATE
        elif h < 4:
            return WaveCondition.ROUGH
        elif h < 6:
            return WaveCondition.VERY_ROUGH
        elif h < 9:
            return WaveCondition.HIGH
        elif h < 14:
            return WaveCondition.VERY_HIGH
        else:
            return WaveCondition.PHENOMENAL


@dataclass 
class WeatherConditions:
    """Current weather conditions data class."""
    temperature_c: Optional[float] = None
    feels_like_c: Optional[float] = None
    humidity_percent: Optional[float] = None
    pressure_hpa: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    wind_gusts_kmh: Optional[float] = None
    precipitation_mm: Optional[float] = None
    weather_code: Optional[int] = None
    weather_description: Optional[str] = None
    visibility_m: Optional[float] = None
    timestamp: Optional[str] = None


@dataclass
class EarthquakeEvent:
    """Earthquake event data class."""
    time: str
    latitude: float
    longitude: float
    depth_km: float
    magnitude: float
    magnitude_type: str
    place: str
    distance_km: float
    tsunami_flag: bool
    tsunami_risk: RiskLevel
    estimated_arrival_minutes: Optional[int] = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in km."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))


def estimate_tsunami_arrival(distance_km: float, depth_m: float = 4000) -> int:
    """Estimate tsunami arrival time in minutes.
    
    Tsunami speed: v = sqrt(g * d) where g=9.81 m/s², d=depth
    Deep ocean: ~800 km/h, Shallow water: ~40-60 km/h
    """
    g = 9.81  # gravity
    speed_ms = sqrt(g * depth_m)  # m/s
    speed_kmh = speed_ms * 3.6  # km/h
    time_hours = distance_km / speed_kmh
    return int(time_hours * 60)


def assess_tsunami_risk(magnitude: float, depth_km: float, distance_km: float) -> Tuple[RiskLevel, str]:
    """Assess tsunami risk based on earthquake parameters.
    
    Returns:
        Tuple of (risk_level, explanation)
    """
    # Tsunamis are typically generated by:
    # - Shallow earthquakes (< 70 km depth)
    # - Large magnitudes (>= 7.0 for significant tsunamis)
    # - Underwater or near-coast epicenters
    
    if magnitude < 6.5:
        return RiskLevel.NONE, "Magnitude too low for tsunami generation"
    
    if depth_km > 100:
        return RiskLevel.LOW, "Deep earthquake - unlikely to generate significant tsunami"
    
    if magnitude >= 8.5 and depth_km < 50:
        if distance_km < 1000:
            return RiskLevel.EXTREME, "Major shallow earthquake nearby - extreme tsunami risk"
        return RiskLevel.HIGH, "Major shallow earthquake - high tsunami risk"
    
    if magnitude >= 7.5 and depth_km < 70:
        if distance_km < 500:
            return RiskLevel.HIGH, "Large shallow earthquake nearby - high tsunami risk"
        return RiskLevel.MODERATE, "Large shallow earthquake - moderate tsunami risk"
    
    if magnitude >= 7.0 and depth_km < 70:
        return RiskLevel.MODERATE, "Moderate earthquake with tsunami potential"
    
    if magnitude >= 6.5:
        return RiskLevel.LOW, "Minor earthquake - low tsunami potential"
    
    return RiskLevel.NONE, "Minimal tsunami risk"


def weather_code_to_description(code: int) -> str:
    """Convert WMO weather code to description."""
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }
    return weather_codes.get(code, "Unknown")


def get_wind_direction_name(degrees: float) -> str:
    """Convert wind direction degrees to compass direction."""
    if degrees is None:
        return "Unknown"
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((degrees + 11.25) / 22.5) % 16
    return directions[idx]


# ============================================================================
# MAIN DATA FETCHER CLASS
# ============================================================================

class KanyakumariOceanMonitor:
    """Comprehensive ocean monitoring system for Kanyakumari region.
    
    This class aggregates data from multiple free APIs to provide:
    - Real-time wave conditions and forecasts
    - Current weather conditions
    - Earthquake monitoring with tsunami risk assessment
    - Official tsunami bulletins
    - Historical tsunami data
    """
    
    def __init__(
        self,
        latitude: float = KANYAKUMARI_LAT,
        longitude: float = KANYAKUMARI_LON,
        location_name: str = "Kanyakumari"
    ):
        """Initialize the ocean monitor.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            location_name: Human-readable location name
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required. Install with: pip install httpx")
        
        self.latitude = latitude
        self.longitude = longitude
        self.location_name = location_name
        self.client = httpx.Client(timeout=30.0)
        self._cache = {}
        self._cache_timeout = timedelta(minutes=5)
        
        LOGGER.info(f"Initialized ocean monitor for {location_name} ({latitude}, {longitude})")
    
    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, 'client'):
            try:
                self.client.close()
            except:
                pass
    
    # ========================================================================
    # MARINE DATA
    # ========================================================================
    
    def fetch_marine_data(self, forecast_hours: int = 72) -> Dict[str, Any]:
        """Fetch marine conditions from Open-Meteo Marine API.
        
        Args:
            forecast_hours: Number of hours to forecast
            
        Returns:
            Dictionary with current conditions, statistics, and forecast
        """
        variables = [
            "wave_height",
            "wave_direction", 
            "wave_period",
            "wind_wave_height",
            "wind_wave_period",
            "wind_wave_direction",
            "swell_wave_height",
            "swell_wave_period",
            "swell_wave_direction",
        ]
        
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ",".join(variables),
            "forecast_hours": forecast_hours,
            "timezone": "Asia/Kolkata",
        }
        
        try:
            response = self.client.get(APIEndpoints.OPEN_METEO_MARINE, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "hourly" not in data:
                LOGGER.warning("No hourly data in marine API response")
                return self._empty_marine_response("No data available")
            
            hourly = data["hourly"]
            times = pd.to_datetime(hourly.get("time", []))
            
            # Build DataFrame
            df = pd.DataFrame({"time": times})
            for var in variables:
                if var in hourly:
                    df[var] = hourly[var]
            
            df.set_index("time", inplace=True)
            df = df.dropna(how='all')
            
            # Get current conditions
            current = self._extract_current_conditions(df)
            
            # Calculate statistics
            stats = self._calculate_wave_statistics(df)
            
            # Find highest wave in forecast
            max_wave = self._find_max_wave(df)
            
            return {
                "success": True,
                "location": {
                    "name": self.location_name,
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                },
                "current": current,
                "statistics": stats,
                "max_wave_forecast": max_wave,
                "forecast_hours": forecast_hours,
                "data_points": len(df),
                "forecast_data": df.reset_index().to_dict(orient="records")[:48],  # First 48 hours
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except httpx.HTTPStatusError as e:
            LOGGER.error(f"HTTP error fetching marine data: {e}")
            return self._empty_marine_response(f"HTTP error: {e.response.status_code}")
        except Exception as e:
            LOGGER.error(f"Error fetching marine data: {e}")
            return self._empty_marine_response(str(e))
    
    def _extract_current_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract current ocean conditions from forecast DataFrame."""
        if df.empty:
            return {}
        
        # Get the row closest to current time
        now = pd.Timestamp.now(tz='Asia/Kolkata').tz_localize(None)
        try:
            current_idx = df.index.get_indexer([now], method='nearest')[0]
            current_row = df.iloc[current_idx]
        except:
            current_row = df.iloc[0]
        
        conditions = OceanConditions(
            wave_height_m=self._safe_float(current_row.get('wave_height')),
            wave_period_s=self._safe_float(current_row.get('wave_period')),
            wave_direction_deg=self._safe_float(current_row.get('wave_direction')),
            swell_height_m=self._safe_float(current_row.get('swell_wave_height')),
            swell_period_s=self._safe_float(current_row.get('swell_wave_period')),
            swell_direction_deg=self._safe_float(current_row.get('swell_wave_direction')),
            wind_wave_height_m=self._safe_float(current_row.get('wind_wave_height')),
            timestamp=current_row.name.isoformat() if hasattr(current_row.name, 'isoformat') else str(current_row.name)
        )
        
        return {
            "wave_height_m": conditions.wave_height_m,
            "wave_period_s": conditions.wave_period_s,
            "wave_direction_deg": conditions.wave_direction_deg,
            "wave_direction_name": get_wind_direction_name(conditions.wave_direction_deg),
            "swell_height_m": conditions.swell_height_m,
            "swell_period_s": conditions.swell_period_s,
            "swell_direction_deg": conditions.swell_direction_deg,
            "wind_wave_height_m": conditions.wind_wave_height_m,
            "wave_condition": conditions.wave_condition.value,
            "wave_condition_display": conditions.wave_condition.name.replace("_", " ").title(),
            "timestamp": conditions.timestamp,
        }
    
    def _calculate_wave_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate wave statistics from forecast data."""
        if df.empty or 'wave_height' not in df.columns:
            return {}
        
        wave_heights = df['wave_height'].dropna()
        
        if wave_heights.empty:
            return {}
        
        return {
            "min_wave_height_m": round(wave_heights.min(), 2),
            "max_wave_height_m": round(wave_heights.max(), 2),
            "mean_wave_height_m": round(wave_heights.mean(), 2),
            "std_wave_height_m": round(wave_heights.std(), 2),
            "significant_wave_height_m": round(wave_heights.quantile(0.67), 2),  # H1/3
            "wave_height_90th_percentile_m": round(wave_heights.quantile(0.90), 2),
        }
    
    def _find_max_wave(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find the maximum wave in the forecast."""
        if df.empty or 'wave_height' not in df.columns:
            return {}
        
        max_idx = df['wave_height'].idxmax()
        if pd.isna(max_idx):
            return {}
        
        max_row = df.loc[max_idx]
        
        return {
            "height_m": self._safe_float(max_row.get('wave_height')),
            "period_s": self._safe_float(max_row.get('wave_period')),
            "direction_deg": self._safe_float(max_row.get('wave_direction')),
            "expected_time": max_idx.isoformat() if hasattr(max_idx, 'isoformat') else str(max_idx),
            "hours_from_now": int((max_idx - pd.Timestamp.now(tz='Asia/Kolkata').tz_localize(None)).total_seconds() / 3600) if hasattr(max_idx, 'total_seconds') else None
        }
    
    def _empty_marine_response(self, error: str = None) -> Dict[str, Any]:
        """Return empty marine response structure."""
        return {
            "success": False,
            "error": error,
            "location": {"name": self.location_name, "latitude": self.latitude, "longitude": self.longitude},
            "current": {},
            "statistics": {},
            "max_wave_forecast": {},
            "forecast_hours": 0,
            "data_points": 0,
            "forecast_data": [],
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    # ========================================================================
    # WEATHER DATA
    # ========================================================================
    
    def fetch_weather_data(self) -> Dict[str, Any]:
        """Fetch current weather conditions from Open-Meteo.
        
        Returns:
            Dictionary with current weather conditions
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "current": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "rain",
                "weather_code",
                "pressure_msl",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m"
            ]),
            "hourly": "temperature_2m,precipitation_probability,weather_code,visibility",
            "forecast_days": 3,
            "timezone": "Asia/Kolkata",
        }
        
        try:
            response = self.client.get(APIEndpoints.OPEN_METEO_WEATHER, params=params)
            response.raise_for_status()
            data = response.json()
            
            current = data.get("current", {})
            weather_code = current.get("weather_code", 0)
            
            conditions = WeatherConditions(
                temperature_c=current.get("temperature_2m"),
                feels_like_c=current.get("apparent_temperature"),
                humidity_percent=current.get("relative_humidity_2m"),
                pressure_hpa=current.get("pressure_msl"),
                wind_speed_kmh=current.get("wind_speed_10m"),
                wind_direction_deg=current.get("wind_direction_10m"),
                wind_gusts_kmh=current.get("wind_gusts_10m"),
                precipitation_mm=current.get("precipitation"),
                weather_code=weather_code,
                weather_description=weather_code_to_description(weather_code),
                timestamp=datetime.utcnow().isoformat()
            )
            
            return {
                "success": True,
                "location": {
                    "name": self.location_name,
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                },
                "temperature_c": conditions.temperature_c,
                "feels_like_c": conditions.feels_like_c,
                "humidity_percent": conditions.humidity_percent,
                "pressure_hpa": conditions.pressure_hpa,
                "wind_speed_kmh": conditions.wind_speed_kmh,
                "wind_direction_deg": conditions.wind_direction_deg,
                "wind_direction_name": get_wind_direction_name(conditions.wind_direction_deg),
                "wind_gusts_kmh": conditions.wind_gusts_kmh,
                "precipitation_mm": conditions.precipitation_mm,
                "weather_code": conditions.weather_code,
                "weather_description": conditions.weather_description,
                "timestamp": conditions.timestamp,
            }
            
        except Exception as e:
            LOGGER.error(f"Error fetching weather data: {e}")
            return {"success": False, "error": str(e)}
    
    # ========================================================================
    # EARTHQUAKE DATA
    # ========================================================================
    
    def fetch_earthquakes(
        self,
        min_magnitude: float = 5.0,
        hours_back: int = 168,  # 7 days
        max_distance_km: float = 5000
    ) -> Dict[str, Any]:
        """Fetch recent earthquakes with tsunami risk assessment.
        
        Args:
            min_magnitude: Minimum earthquake magnitude
            hours_back: Hours of history to search
            max_distance_km: Maximum distance from location
            
        Returns:
            Dictionary with earthquake events and tsunami risk analysis
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        # Search Indian Ocean region
        params = {
            "format": "geojson",
            "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "minmagnitude": min_magnitude,
            "limit": 100,
            "orderby": "time-asc",
            "minlatitude": -15,
            "maxlatitude": 30,
            "minlongitude": 40,
            "maxlongitude": 120,
        }
        
        try:
            response = self.client.get(APIEndpoints.USGS_EARTHQUAKE, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "features" not in data:
                return self._empty_earthquake_response()
            
            events = []
            tsunami_risk_events = []
            
            for feature in data["features"]:
                props = feature["properties"]
                coords = feature["geometry"]["coordinates"]
                
                eq_lat, eq_lon, eq_depth = coords[1], coords[0], coords[2]
                distance = haversine_distance(self.latitude, self.longitude, eq_lat, eq_lon)
                
                if distance > max_distance_km:
                    continue
                
                magnitude = props.get("mag", 0)
                risk_level, risk_reason = assess_tsunami_risk(magnitude, eq_depth, distance)
                
                event = {
                    "time": datetime.fromtimestamp(props.get("time", 0) / 1000).isoformat(),
                    "latitude": eq_lat,
                    "longitude": eq_lon,
                    "depth_km": eq_depth,
                    "magnitude": magnitude,
                    "magnitude_type": props.get("magType", ""),
                    "place": props.get("place", ""),
                    "usgs_tsunami_flag": bool(props.get("tsunami", 0)),
                    "distance_km": round(distance, 1),
                    "tsunami_risk_level": risk_level.value,
                    "tsunami_risk_reason": risk_reason,
                    "estimated_arrival_minutes": estimate_tsunami_arrival(distance) if risk_level != RiskLevel.NONE else None,
                }
                
                events.append(event)
                
                if risk_level in [RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.EXTREME]:
                    tsunami_risk_events.append(event)
            
            # Sort by magnitude (highest first)
            events.sort(key=lambda x: x["magnitude"] or 0, reverse=True)
            
            return {
                "success": True,
                "location": {"name": self.location_name, "latitude": self.latitude, "longitude": self.longitude},
                "search_period_hours": hours_back,
                "min_magnitude": min_magnitude,
                "total_count": len(events),
                "tsunami_risk_count": len(tsunami_risk_events),
                "events": events,
                "tsunami_risk_events": tsunami_risk_events,
                "highest_magnitude": max((e["magnitude"] for e in events), default=None) if events else None,
                "nearest_earthquake_km": min((e["distance_km"] for e in events), default=None) if events else None,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            LOGGER.error(f"Error fetching earthquake data: {e}")
            return self._empty_earthquake_response(str(e))
    
    def _empty_earthquake_response(self, error: str = None) -> Dict[str, Any]:
        """Return empty earthquake response structure."""
        return {
            "success": False,
            "error": error,
            "events": [],
            "tsunami_risk_events": [],
            "total_count": 0,
            "tsunami_risk_count": 0,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    # ========================================================================
    # TSUNAMI BULLETINS
    # ========================================================================
    
    def fetch_tsunami_bulletins(self) -> Dict[str, Any]:
        """Fetch official tsunami bulletins from NOAA warning centers.
        
        Returns:
            Dictionary with active bulletins and warnings
        """
        if not FEEDPARSER_AVAILABLE:
            return {
                "success": False,
                "error": "feedparser not installed. Install with: pip install feedparser",
                "bulletins": [],
            }
        
        bulletins = []
        sources_checked = []
        
        for url, source_name in [
            (APIEndpoints.PTWC_RSS, "Pacific Tsunami Warning Center"),
            (APIEndpoints.NTWC_RSS, "National Tsunami Warning Center"),
        ]:
            try:
                feed = feedparser.parse(url)
                sources_checked.append({"name": source_name, "status": "success"})
                
                for entry in feed.entries:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")[:1000]
                    
                    # Determine severity from title
                    severity = "INFORMATION"
                    if any(word in title.upper() for word in ["WARNING", "THREAT"]):
                        severity = "WARNING"
                    elif any(word in title.upper() for word in ["WATCH", "ADVISORY"]):
                        severity = "WATCH"
                    elif "CANCEL" in title.upper():
                        severity = "CANCELLED"
                    
                    # Check if it might affect Indian Ocean
                    affects_region = any(
                        region in (title + summary).upper()
                        for region in ["INDIAN", "INDIA", "INDONESIA", "SRI LANKA", "MALDIVES", "THAILAND", "ANDAMAN"]
                    )
                    
                    bulletin = {
                        "title": title,
                        "summary": summary,
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": source_name,
                        "severity": severity,
                        "affects_indian_ocean_region": affects_region,
                    }
                    bulletins.append(bulletin)
                    
            except Exception as e:
                LOGGER.warning(f"Failed to fetch {source_name} bulletins: {e}")
                sources_checked.append({"name": source_name, "status": "error", "error": str(e)})
        
        # Filter active warnings
        active_warnings = [
            b for b in bulletins
            if b["severity"] in ["WARNING", "WATCH"] and b.get("affects_indian_ocean_region", False)
        ]
        
        return {
            "success": True,
            "location": {"name": self.location_name, "latitude": self.latitude, "longitude": self.longitude},
            "total_bulletins": len(bulletins),
            "bulletins": bulletins[:20],  # Limit to 20 most recent
            "active_warnings": active_warnings,
            "active_warning_count": len(active_warnings),
            "sources_checked": sources_checked,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    # ========================================================================
    # HISTORICAL DATA
    # ========================================================================
    
    def get_historical_tsunamis(self) -> Dict[str, Any]:
        """Get historical tsunami data for the region.
        
        Returns:
            Dictionary with historical tsunami events
        """
        return {
            "success": True,
            "location": {"name": self.location_name, "latitude": self.latitude, "longitude": self.longitude},
            "historical_tsunamis": HISTORICAL_TSUNAMIS,
            "total_recorded": len(HISTORICAL_TSUNAMIS),
            "most_severe": HISTORICAL_TSUNAMIS[0] if HISTORICAL_TSUNAMIS else None,
            "years_since_major_event": (datetime.now().year - 2004) if HISTORICAL_TSUNAMIS else None,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    # ========================================================================
    # COMBINED DATA FETCH
    # ========================================================================
    
    def fetch_all_data(self) -> Dict[str, Any]:
        """Fetch all available data sources.
        
        Returns:
            Comprehensive dictionary with all ocean monitoring data
        """
        marine_data = self.fetch_marine_data()
        weather_data = self.fetch_weather_data()
        earthquake_data = self.fetch_earthquakes()
        bulletin_data = self.fetch_tsunami_bulletins()
        historical_data = self.get_historical_tsunamis()
        
        # Calculate overall risk assessment
        overall_risk = self._calculate_overall_risk(marine_data, earthquake_data, bulletin_data)
        
        return {
            "success": True,
            "location": {
                "name": self.location_name,
                "latitude": self.latitude,
                "longitude": self.longitude,
                "region": KANYAKUMARI_REGION["name"],
            },
            "marine": marine_data,
            "weather": weather_data,
            "earthquakes": earthquake_data,
            "bulletins": bulletin_data,
            "historical": historical_data,
            "overall_risk_assessment": overall_risk,
            "data_sources": {
                "marine_api": "Open-Meteo Marine",
                "weather_api": "Open-Meteo Weather",
                "earthquake_api": "USGS Earthquake Catalog",
                "bulletin_sources": ["PTWC", "NTWC"],
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def _calculate_overall_risk(
        self,
        marine: Dict,
        earthquakes: Dict,
        bulletins: Dict
    ) -> Dict[str, Any]:
        """Calculate overall risk assessment from all data sources."""
        
        risk_score = 0.0
        risk_factors = []
        
        # Wave conditions factor
        if marine.get("success") and marine.get("current"):
            wave_height = marine["current"].get("wave_height_m", 0) or 0
            if wave_height > 4:
                risk_score += 0.3
                risk_factors.append(f"High waves: {wave_height:.1f}m")
            elif wave_height > 2.5:
                risk_score += 0.1
                risk_factors.append(f"Rough seas: {wave_height:.1f}m")
        
        # Earthquake/tsunami factor
        if earthquakes.get("success"):
            tsunami_events = earthquakes.get("tsunami_risk_events", [])
            if tsunami_events:
                highest_risk = max(
                    (RiskLevel[e["tsunami_risk_level"].upper()] for e in tsunami_events),
                    key=lambda x: ["none", "low", "moderate", "high", "extreme"].index(x.value),
                    default=RiskLevel.NONE
                )
                if highest_risk == RiskLevel.EXTREME:
                    risk_score += 0.5
                    risk_factors.append("EXTREME tsunami risk from recent earthquake")
                elif highest_risk == RiskLevel.HIGH:
                    risk_score += 0.3
                    risk_factors.append("HIGH tsunami risk from recent earthquake")
                elif highest_risk == RiskLevel.MODERATE:
                    risk_score += 0.15
                    risk_factors.append("Moderate tsunami risk from recent earthquake")
        
        # Active warnings factor
        if bulletins.get("success"):
            active_warnings = bulletins.get("active_warning_count", 0)
            if active_warnings > 0:
                risk_score += 0.4
                risk_factors.append(f"{active_warnings} active tsunami warning(s)")
        
        # Determine overall level
        risk_score = min(risk_score, 1.0)
        
        if risk_score >= 0.7:
            overall_level = RiskLevel.EXTREME
        elif risk_score >= 0.5:
            overall_level = RiskLevel.HIGH
        elif risk_score >= 0.3:
            overall_level = RiskLevel.MODERATE
        elif risk_score >= 0.1:
            overall_level = RiskLevel.LOW
        else:
            overall_level = RiskLevel.NONE
        
        return {
            "risk_level": overall_level.value,
            "risk_score": round(risk_score, 2),
            "risk_factors": risk_factors,
            "recommendation": self._get_risk_recommendation(overall_level),
        }
    
    def _get_risk_recommendation(self, level: RiskLevel) -> str:
        """Get safety recommendation based on risk level."""
        recommendations = {
            RiskLevel.NONE: "Normal conditions. Safe for regular coastal activities.",
            RiskLevel.LOW: "Minor elevated risk. Stay informed about weather updates.",
            RiskLevel.MODERATE: "Caution advised. Avoid swimming in rough waters. Monitor official alerts.",
            RiskLevel.HIGH: "HIGH ALERT. Avoid coastal areas. Prepare for possible evacuation.",
            RiskLevel.EXTREME: "EMERGENCY. Evacuate coastal areas immediately. Follow official instructions.",
        }
        return recommendations.get(level, "Stay alert and monitor conditions.")
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def prepare_model_input(self, hours: int = 24) -> Tuple[np.ndarray, Dict]:
        """Prepare data for model input.
        
        Fetches marine data and formats it for the CNN-LSTM model.
        
        Args:
            hours: Number of hours of data to prepare
            
        Returns:
            Tuple of (sequence_array, metadata)
        """
        marine_data = self.fetch_marine_data(forecast_hours=hours)
        
        if not marine_data.get("success") or not marine_data.get("forecast_data"):
            # Return zeros if no data
            return np.zeros((hours, 8)), {"success": False, "error": "No data available"}
        
        # Extract features from forecast data
        features = []
        feature_names = [
            'wave_height', 'wave_period', 'wave_direction',
            'swell_wave_height', 'swell_wave_period',
            'wind_wave_height', 'wind_wave_period', 'wind_wave_direction'
        ]
        
        for record in marine_data["forecast_data"][:hours]:
            row = []
            for feat in feature_names:
                value = record.get(feat)
                row.append(value if value is not None and not np.isnan(value) else 0.0)
            features.append(row)
        
        # Pad if necessary
        while len(features) < hours:
            features.append([0.0] * len(feature_names))
        
        sequence = np.array(features[:hours], dtype=np.float32)
        
        return sequence, {
            "success": True,
            "feature_names": feature_names,
            "data_points": len(marine_data["forecast_data"]),
            "timestamp": marine_data["timestamp"],
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_kanyakumari_data() -> Dict[str, Any]:
    """Quick function to fetch all Kanyakumari ocean data.
    
    Returns:
        Comprehensive ocean monitoring data dictionary
    """
    monitor = KanyakumariOceanMonitor()
    return monitor.fetch_all_data()


def get_current_conditions() -> Dict[str, Any]:
    """Get current ocean and weather conditions for Kanyakumari.
    
    Returns:
        Dictionary with current conditions
    """
    monitor = KanyakumariOceanMonitor()
    return {
        "marine": monitor.fetch_marine_data(forecast_hours=24),
        "weather": monitor.fetch_weather_data(),
        "location": {
            "name": "Kanyakumari",
            "latitude": KANYAKUMARI_LAT,
            "longitude": KANYAKUMARI_LON,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    # Test the data fetcher
    print("=" * 80)
    print("KANYAKUMARI OCEAN MONITORING SYSTEM")
    print("=" * 80)
    
    monitor = KanyakumariOceanMonitor()
    data = monitor.fetch_all_data()
    
    print(f"\n📍 Location: {data['location']['name']}")
    print(f"   Coordinates: {data['location']['latitude']}, {data['location']['longitude']}")
    
    if data.get('marine', {}).get('success'):
        marine = data['marine']
        current = marine.get('current', {})
        print(f"\n🌊 Current Wave Conditions:")
        print(f"   Wave Height: {current.get('wave_height_m', 'N/A')} m")
        print(f"   Wave Period: {current.get('wave_period_s', 'N/A')} s")
        print(f"   Condition: {current.get('wave_condition_display', 'N/A')}")
    
    if data.get('weather', {}).get('success'):
        weather = data['weather']
        print(f"\n☀️ Weather Conditions:")
        print(f"   Temperature: {weather.get('temperature_c', 'N/A')}°C")
        print(f"   Wind: {weather.get('wind_speed_kmh', 'N/A')} km/h {weather.get('wind_direction_name', '')}")
        print(f"   Conditions: {weather.get('weather_description', 'N/A')}")
    
    risk = data.get('overall_risk_assessment', {})
    print(f"\n⚠️ Overall Risk Assessment:")
    print(f"   Level: {risk.get('risk_level', 'N/A').upper()}")
    print(f"   Score: {risk.get('risk_score', 0):.2f}")
    print(f"   Recommendation: {risk.get('recommendation', 'N/A')}")
    
    print("\n" + "=" * 80)
