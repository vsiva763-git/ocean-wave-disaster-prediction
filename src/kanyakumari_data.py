"""Real-time data fetcher for Kanyakumari ocean monitoring.

Aggregates data from multiple free APIs:
- Open-Meteo Marine API (wave forecasts)
- USGS Earthquake Catalog (seismic events)
- NOAA Tsunami Bulletins (official warnings)
- Historical tsunami data for the region
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from math import radians, sin, cos, asin, sqrt

import numpy as np
import pandas as pd

try:
    import httpx
except ImportError:
    httpx = None

try:
    import feedparser
except ImportError:
    feedparser = None

from .config import (
    KANYAKUMARI_LAT, 
    KANYAKUMARI_LON, 
    KANYAKUMARI_REGION,
    HISTORICAL_TSUNAMIS,
)

LOGGER = logging.getLogger(__name__)

# API URLs
OPEN_METEO_MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
USGS_EARTHQUAKE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
PTWC_RSS_URL = "https://www.tsunami.gov/events/xml/PHEBxml.xml"
NTWC_RSS_URL = "https://www.tsunami.gov/events/xml/WCEBxml.xml"

# INCOIS RSS for Indian Ocean (if available)
INCOIS_URL = "https://incois.gov.in"  # Placeholder - would need proper API


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))


class KanyakumariDataFetcher:
    """Unified data fetcher for Kanyakumari ocean monitoring."""
    
    def __init__(
        self,
        latitude: float = KANYAKUMARI_LAT,
        longitude: float = KANYAKUMARI_LON,
    ):
        if httpx is None:
            raise ImportError("httpx is required. Install with: pip install httpx")
        
        self.latitude = latitude
        self.longitude = longitude
        self.client = httpx.Client(timeout=30.0)
        
    def __del__(self):
        if hasattr(self, 'client'):
            self.client.close()
    
    def fetch_marine_data(self, hours: int = 48) -> Dict[str, Any]:
        """Fetch marine forecast data from Open-Meteo.
        
        Args:
            hours: Number of forecast hours
            
        Returns:
            Dictionary with current conditions and forecast data
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
            "forecast_hours": hours,
            "timezone": "Asia/Kolkata",
        }
        
        try:
            response = self.client.get(OPEN_METEO_MARINE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "hourly" not in data:
                LOGGER.warning("No hourly data from Open-Meteo Marine API")
                return self._empty_marine_response()
            
            hourly = data["hourly"]
            times = pd.to_datetime(hourly["time"])
            
            # Create DataFrame
            df = pd.DataFrame({"time": times})
            for var in variables:
                if var in hourly:
                    df[var] = hourly[var]
            
            df.set_index("time", inplace=True)
            
            # Extract current conditions (most recent data point with values)
            current = self._get_current_conditions(df)
            
            # Calculate statistics
            stats = self._calculate_wave_statistics(df)
            
            return {
                "success": True,
                "location": {
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    "name": "Kanyakumari" if self._is_kanyakumari() else "Custom Location",
                },
                "current": current,
                "statistics": stats,
                "forecast_hours": hours,
                "data_points": len(df),
                "forecast": df.to_dict(orient="records"),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as exc:
            LOGGER.error(f"Failed to fetch marine data: {exc}")
            return self._empty_marine_response(str(exc))
    
    def fetch_weather_data(self) -> Dict[str, Any]:
        """Fetch current weather conditions from Open-Meteo."""
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,weather_code,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
            "hourly": "temperature_2m,precipitation_probability,weather_code,visibility",
            "forecast_days": 3,
            "timezone": "Asia/Kolkata",
        }
        
        try:
            response = self.client.get(OPEN_METEO_WEATHER_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            current = data.get("current", {})
            
            # Map weather codes to descriptions
            weather_code = current.get("weather_code", 0)
            weather_desc = self._weather_code_to_description(weather_code)
            
            return {
                "success": True,
                "temperature_c": current.get("temperature_2m"),
                "feels_like_c": current.get("apparent_temperature"),
                "humidity_percent": current.get("relative_humidity_2m"),
                "pressure_hpa": current.get("pressure_msl"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "wind_direction": current.get("wind_direction_10m"),
                "wind_gusts_kmh": current.get("wind_gusts_10m"),
                "precipitation_mm": current.get("precipitation"),
                "weather_code": weather_code,
                "weather_description": weather_desc,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as exc:
            LOGGER.error(f"Failed to fetch weather data: {exc}")
            return {"success": False, "error": str(exc)}
    
    def fetch_earthquakes(
        self,
        min_magnitude: float = 5.0,
        hours_back: int = 72,
        max_distance_km: float = 3000,
    ) -> Dict[str, Any]:
        """Fetch recent earthquakes that could affect Kanyakumari.
        
        Args:
            min_magnitude: Minimum earthquake magnitude
            hours_back: Hours of history to fetch
            max_distance_km: Maximum distance from location to consider
            
        Returns:
            Dictionary with earthquake events and tsunami risk assessment
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        # Focus on Indian Ocean region for tsunami-generating earthquakes
        params = {
            "format": "geojson",
            "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "minmagnitude": min_magnitude,
            "limit": 100,
            "orderby": "time-asc",
            # Indian Ocean region bounding box
            "minlatitude": -10,
            "maxlatitude": 25,
            "minlongitude": 50,
            "maxlongitude": 100,
        }
        
        try:
            response = self.client.get(USGS_EARTHQUAKE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "features" not in data or not data["features"]:
                return {
                    "success": True,
                    "count": 0,
                    "events": [],
                    "tsunami_risk_events": [],
                    "highest_magnitude": None,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            
            events = []
            tsunami_risk_events = []
            
            for feature in data["features"]:
                props = feature["properties"]
                coords = feature["geometry"]["coordinates"]
                
                eq_lat, eq_lon = coords[1], coords[0]
                distance = haversine_distance(self.latitude, self.longitude, eq_lat, eq_lon)
                
                if distance > max_distance_km:
                    continue
                
                event = {
                    "time": datetime.fromtimestamp(props.get("time", 0) / 1000).isoformat(),
                    "latitude": eq_lat,
                    "longitude": eq_lon,
                    "depth_km": coords[2],
                    "magnitude": props.get("mag"),
                    "magnitude_type": props.get("magType", ""),
                    "place": props.get("place", ""),
                    "tsunami_flag": props.get("tsunami", 0),
                    "distance_km": round(distance, 1),
                    "estimated_arrival_minutes": self._estimate_tsunami_arrival(distance),
                }
                
                # Assess tsunami risk
                risk = self._assess_tsunami_risk(event)
                event["tsunami_risk"] = risk
                
                events.append(event)
                
                if risk["level"] in ["MODERATE", "HIGH", "EXTREME"]:
                    tsunami_risk_events.append(event)
            
            # Sort by magnitude (highest first)
            events.sort(key=lambda x: x["magnitude"] or 0, reverse=True)
            
            return {
                "success": True,
                "count": len(events),
                "events": events,
                "tsunami_risk_events": tsunami_risk_events,
                "highest_magnitude": max((e["magnitude"] for e in events), default=None),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as exc:
            LOGGER.error(f"Failed to fetch earthquake data: {exc}")
            return {"success": False, "error": str(exc), "events": []}
    
    def fetch_tsunami_bulletins(self) -> Dict[str, Any]:
        """Fetch official tsunami bulletins from NOAA warning centers."""
        if feedparser is None:
            return {
                "success": False, 
                "error": "feedparser not installed",
                "bulletins": [],
            }
        
        bulletins = []
        
        for url, source in [(PTWC_RSS_URL, "PTWC"), (NTWC_RSS_URL, "NTWC")]:
            try:
                feed = feedparser.parse(url)
                
                for entry in feed.entries:
                    bulletin = {
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", "")[:500],  # Truncate
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": source,
                        "severity": self._parse_bulletin_severity(entry.get("title", "")),
                        "affects_indian_ocean": self._affects_indian_ocean(entry),
                    }
                    bulletins.append(bulletin)
                    
            except Exception as exc:
                LOGGER.warning(f"Failed to fetch {source} bulletins: {exc}")
        
        # Sort by recency
        bulletins.sort(key=lambda x: x["published"], reverse=True)
        
        return {
            "success": True,
            "count": len(bulletins),
            "bulletins": bulletins,
            "active_warnings": [b for b in bulletins if b["severity"] in ["WARNING", "WATCH"]],
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def fetch_historical_data(self) -> Dict[str, Any]:
        """Get historical tsunami data for Kanyakumari region."""
        return {
            "success": True,
            "historical_tsunamis": HISTORICAL_TSUNAMIS,
            "last_major_event": HISTORICAL_TSUNAMIS[0] if HISTORICAL_TSUNAMIS else None,
            "total_recorded_events": len(HISTORICAL_TSUNAMIS),
        }
    
    def fetch_all_data(self) -> Dict[str, Any]:
        """Fetch all available data sources."""
        return {
            "marine": self.fetch_marine_data(),
            "weather": self.fetch_weather_data(),
            "earthquakes": self.fetch_earthquakes(),
            "bulletins": self.fetch_tsunami_bulletins(),
            "historical": self.fetch_historical_data(),
            "location": {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "name": "Kanyakumari" if self._is_kanyakumari() else "Custom Location",
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    # Helper methods
    def _is_kanyakumari(self) -> bool:
        """Check if location is near Kanyakumari."""
        return haversine_distance(
            self.latitude, self.longitude,
            KANYAKUMARI_LAT, KANYAKUMARI_LON
        ) < 50  # Within 50 km
    
    def _get_current_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract current conditions from forecast DataFrame."""
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        
        # Find closest time to now
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        closest_idx = (df.index - now).abs().argmin()
        current_row = df.iloc[closest_idx]
        
        wave_height = current_row.get("wave_height")
        max_wave = df["wave_height"].max() if "wave_height" in df.columns else None
        
        return {
            "wave_height_m": round(wave_height, 2) if pd.notna(wave_height) else None,
            "wave_period_s": round(current_row.get("wave_period", 0), 1) if pd.notna(current_row.get("wave_period")) else None,
            "wave_direction_deg": round(current_row.get("wave_direction", 0), 0) if pd.notna(current_row.get("wave_direction")) else None,
            "swell_height_m": round(current_row.get("swell_wave_height", 0), 2) if pd.notna(current_row.get("swell_wave_height")) else None,
            "wind_wave_height_m": round(current_row.get("wind_wave_height", 0), 2) if pd.notna(current_row.get("wind_wave_height")) else None,
            "max_wave_height_48h": round(max_wave, 2) if pd.notna(max_wave) else None,
            "time": df.index[closest_idx].isoformat(),
        }
    
    def _calculate_wave_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate wave statistics from forecast data."""
        if "wave_height" not in df.columns:
            return {}
        
        heights = df["wave_height"].dropna()
        
        if heights.empty:
            return {}
        
        return {
            "mean_wave_height": round(heights.mean(), 2),
            "max_wave_height": round(heights.max(), 2),
            "min_wave_height": round(heights.min(), 2),
            "std_wave_height": round(heights.std(), 2),
            "wave_trend": self._calculate_trend(heights),
        }
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate if values are trending up, down, or stable."""
        if len(series) < 2:
            return "STABLE"
        
        first_half = series[:len(series)//2].mean()
        second_half = series[len(series)//2:].mean()
        
        diff = second_half - first_half
        if diff > 0.5:
            return "INCREASING"
        elif diff < -0.5:
            return "DECREASING"
        return "STABLE"
    
    def _estimate_tsunami_arrival(self, distance_km: float) -> int:
        """Estimate tsunami arrival time in minutes based on distance."""
        # Tsunami speed in open ocean: ~700-800 km/h
        # Speed decreases in shallow water
        avg_speed_kmh = 600  # Conservative estimate
        return int((distance_km / avg_speed_kmh) * 60)
    
    def _assess_tsunami_risk(self, event: Dict) -> Dict[str, Any]:
        """Assess tsunami risk from an earthquake event."""
        magnitude = event.get("magnitude", 0) or 0
        depth = event.get("depth_km", 100) or 100
        distance = event.get("distance_km", 10000) or 10000
        tsunami_flag = event.get("tsunami_flag", 0)
        
        risk_score = 0
        factors = []
        
        # Magnitude factor (most important)
        if magnitude >= 9.0:
            risk_score += 50
            factors.append(f"Extreme magnitude ({magnitude})")
        elif magnitude >= 8.0:
            risk_score += 40
            factors.append(f"Major magnitude ({magnitude})")
        elif magnitude >= 7.0:
            risk_score += 25
            factors.append(f"Strong magnitude ({magnitude})")
        elif magnitude >= 6.5:
            risk_score += 10
            factors.append(f"Moderate magnitude ({magnitude})")
        
        # Depth factor (shallow = worse)
        if depth < 30:
            risk_score += 20
            factors.append(f"Shallow depth ({depth} km)")
        elif depth < 70:
            risk_score += 10
            factors.append(f"Moderate depth ({depth} km)")
        
        # Distance factor
        if distance < 500:
            risk_score += 15
            factors.append(f"Close proximity ({distance} km)")
        elif distance < 1000:
            risk_score += 10
        
        # Official tsunami flag
        if tsunami_flag:
            risk_score += 20
            factors.append("Official tsunami flag raised")
        
        # Determine risk level
        if risk_score >= 60:
            level = "EXTREME"
        elif risk_score >= 40:
            level = "HIGH"
        elif risk_score >= 20:
            level = "MODERATE"
        else:
            level = "LOW"
        
        return {
            "level": level,
            "score": risk_score,
            "factors": factors,
        }
    
    def _parse_bulletin_severity(self, title: str) -> str:
        """Parse severity level from bulletin title."""
        title_upper = title.upper()
        if "WARNING" in title_upper:
            return "WARNING"
        elif "WATCH" in title_upper:
            return "WATCH"
        elif "ADVISORY" in title_upper:
            return "ADVISORY"
        elif "INFORMATION" in title_upper or "STATEMENT" in title_upper:
            return "INFORMATION"
        return "UNKNOWN"
    
    def _affects_indian_ocean(self, entry: Dict) -> bool:
        """Check if bulletin affects Indian Ocean region."""
        text = (entry.get("title", "") + " " + entry.get("summary", "")).upper()
        keywords = ["INDIAN OCEAN", "INDIA", "SRI LANKA", "INDONESIA", "THAILAND", 
                   "ANDAMAN", "NICOBAR", "BAY OF BENGAL", "ARABIAN SEA"]
        return any(kw in text for kw in keywords)
    
    def _weather_code_to_description(self, code: int) -> str:
        """Convert WMO weather code to description."""
        codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
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
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail",
        }
        return codes.get(code, "Unknown")
    
    def _empty_marine_response(self, error: str = None) -> Dict[str, Any]:
        """Return empty marine response structure."""
        return {
            "success": False,
            "error": error,
            "location": {
                "latitude": self.latitude,
                "longitude": self.longitude,
            },
            "current": {},
            "statistics": {},
            "forecast": [],
            "timestamp": datetime.utcnow().isoformat(),
        }


# Convenience function for quick data fetch
def fetch_kanyakumari_data() -> Dict[str, Any]:
    """Fetch all ocean data for Kanyakumari."""
    fetcher = KanyakumariDataFetcher()
    return fetcher.fetch_all_data()


def fetch_location_data(latitude: float, longitude: float) -> Dict[str, Any]:
    """Fetch all ocean data for a custom location."""
    fetcher = KanyakumariDataFetcher(latitude, longitude)
    return fetcher.fetch_all_data()
