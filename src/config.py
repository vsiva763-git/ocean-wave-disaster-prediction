"""Configuration for Ocean Wave Disaster Prediction System.

Focused on Kanyakumari region - the southernmost tip of India where
the Arabian Sea, Bay of Bengal, and Indian Ocean meet.
"""
from dataclasses import dataclass
from typing import List, Tuple

# Kanyakumari Location - Southernmost tip of India
KANYAKUMARI_LAT = 8.0883
KANYAKUMARI_LON = 77.5385

# Region bounding box for data fetching (covers surrounding ocean)
KANYAKUMARI_REGION = {
    "name": "Kanyakumari",
    "center_lat": KANYAKUMARI_LAT,
    "center_lon": KANYAKUMARI_LON,
    "min_lat": 5.0,   # Extended south into Indian Ocean
    "max_lat": 12.0,  # Northern limit
    "min_lon": 74.0,  # Arabian Sea side
    "max_lon": 82.0,  # Bay of Bengal side
}

# Nearby NDBC-equivalent stations and buoys
# Note: Indian Ocean buoys are managed by INCOIS (Indian National Centre for Ocean Information Services)
NEARBY_STATIONS = {
    "BD08": {"lat": 8.3, "lon": 73.2, "name": "Lakshadweep Sea"},
    "BD09": {"lat": 6.0, "lon": 81.0, "name": "Sri Lanka Southwest"},
    "BD11": {"lat": 10.5, "lon": 72.0, "name": "Arabian Sea Central"},
    "23223": {"lat": 6.0, "lon": 90.0, "name": "Bay of Bengal South"},
}

# Historical Tsunamis that affected Kanyakumari
HISTORICAL_TSUNAMIS = [
    {
        "date": "2004-12-26",
        "name": "Indian Ocean Tsunami",
        "magnitude": 9.1,
        "epicenter_lat": 3.316,
        "epicenter_lon": 95.854,
        "deaths_kanyakumari": 802,
        "max_wave_height_m": 10.0,
        "arrival_time_minutes": 120,
    },
    {
        "date": "1883-08-27",
        "name": "Krakatoa Tsunami",
        "magnitude": 6.0,  # Volcanic
        "epicenter_lat": -6.102,
        "epicenter_lon": 105.423,
        "max_wave_height_m": 1.5,
    },
]


@dataclass
class ModelConfig:
    """Configuration for the CNN-LSTM hybrid model."""
    # Image input (for satellite/radar imagery)
    image_size: Tuple[int, int] = (128, 128)
    image_channels: int = 3
    
    # Sequence input (time series data)
    seq_len: int = 24  # 24 hours of hourly data
    seq_features: int = 8  # Number of ocean/weather parameters
    
    # Feature columns for time series
    feature_columns: List[str] = None
    
    # Model architecture
    cnn_backbone: str = "simple"  # "simple" or "mobilenet_v2"
    lstm_units: List[int] = None
    dense_units: List[int] = None
    dropout_rate: float = 0.3
    
    # Output classes
    num_classes: int = 4  # NORMAL, MODERATE, HIGH_RISK, TSUNAMI_WARNING
    class_names: List[str] = None
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                "wave_height",          # Significant wave height (m)
                "wave_period",          # Wave period (seconds)
                "wave_direction",       # Wave direction (degrees)
                "wind_speed",           # Wind speed (km/h)
                "wind_direction",       # Wind direction (degrees)
                "sea_surface_temp",     # Sea surface temperature (°C)
                "swell_height",         # Swell wave height (m)
                "pressure",             # Atmospheric pressure (hPa)
            ]
        
        if self.lstm_units is None:
            self.lstm_units = [128, 64]
        
        if self.dense_units is None:
            self.dense_units = [128, 64]
        
        if self.class_names is None:
            self.class_names = ["NORMAL", "MODERATE", "HIGH_RISK", "TSUNAMI_WARNING"]


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Data refresh intervals (seconds)
    marine_data_refresh: int = 3600  # 1 hour
    earthquake_refresh: int = 300    # 5 minutes
    bulletin_refresh: int = 600      # 10 minutes
    
    # Alert thresholds
    wave_height_moderate: float = 2.5   # meters
    wave_height_high: float = 4.0       # meters
    wave_height_extreme: float = 6.0    # meters
    earthquake_tsunami_min_magnitude: float = 6.5


# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_API_CONFIG = APIConfig()
