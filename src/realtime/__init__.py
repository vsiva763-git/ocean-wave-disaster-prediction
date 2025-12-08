"""Real-time data fetching and monitoring for ocean wave disaster prediction.

This module provides utilities to fetch live oceanographic data from various
free APIs and prepare them for model inference.
"""

from .open_meteo import fetch_open_meteo_marine
from .ndbc import fetch_ndbc_latest
from .usgs_earthquake import fetch_usgs_earthquakes
from .tsunami_bulletins import fetch_tsunami_bulletins

__all__ = [
    "fetch_open_meteo_marine",
    "fetch_ndbc_latest",
    "fetch_usgs_earthquakes",
    "fetch_tsunami_bulletins",
]
