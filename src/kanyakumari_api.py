"""Kanyakumari Ocean Wave & Tsunami Prediction API.

FastAPI service providing:
- Real-time ocean wave predictions using CNN-LSTM hybrid model
- Tsunami risk assessment based on seismic activity
- Live marine and weather data for Kanyakumari coast
- Comprehensive web interface for monitoring

Author: Ocean Wave Disaster Prediction System
Version: 2.0.0
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add src to path
SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    raise ImportError("FastAPI required. Install: pip install fastapi uvicorn")

try:
    from kanyakumari_monitor import (
        KanyakumariOceanMonitor,
        KANYAKUMARI_LAT,
        KANYAKUMARI_LON,
        KANYAKUMARI_REGION,
        HISTORICAL_TSUNAMIS,
        MONITORING_POINTS,
    )
except ImportError as e:
    print(f"Import error: {e}")
    raise

try:
    from models.hybrid_cnn_lstm import OceanWavePredictor, build_multimodal_hybrid_model
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    OceanWavePredictor = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOGGER = logging.getLogger(__name__)

# Paths
WEB_DIR = PROJECT_ROOT / "web"
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"
MODELS_DIR = PROJECT_ROOT / "models"

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="🌊 Kanyakumari Ocean Wave & Tsunami Prediction System",
    description="""
    Real-time ocean monitoring and disaster prediction for Kanyakumari coastal region.
    
    ## Features
    - **Live Wave Prediction**: CNN-LSTM hybrid model for wave height and severity prediction
    - **Tsunami Risk Assessment**: Real-time earthquake monitoring with tsunami risk analysis
    - **Weather Monitoring**: Current atmospheric conditions affecting sea state
    - **Official Bulletins**: Aggregated tsunami warnings from NOAA centers
    
    ## Data Sources
    - Open-Meteo Marine API (wave forecasts)
    - Open-Meteo Weather API (atmospheric conditions)
    - USGS Earthquake Catalog (seismic events)
    - NOAA Tsunami Warning Centers (official bulletins)
    
    ## Location
    Kanyakumari (8.0883°N, 77.5385°E) - The southernmost tip of India where 
    the Arabian Sea, Bay of Bengal, and Indian Ocean converge.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    LOGGER.warning(f"Static directory not found: {STATIC_DIR}")

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Global predictor instance
predictor: Optional[OceanWavePredictor] = None
monitor: Optional[KanyakumariOceanMonitor] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global predictor, monitor
    
    LOGGER.info("=" * 60)
    LOGGER.info("🌊 Starting Kanyakumari Ocean Wave Prediction System")
    LOGGER.info("=" * 60)
    
    # Initialize ocean monitor
    try:
        monitor = KanyakumariOceanMonitor()
        LOGGER.info(f"✅ Ocean monitor initialized for {monitor.location_name}")
    except Exception as e:
        LOGGER.error(f"❌ Failed to initialize ocean monitor: {e}")
    
    # Initialize predictor (with mock if no trained model)
    if MODEL_AVAILABLE:
        try:
            predictor = OceanWavePredictor()
            LOGGER.info("✅ Wave predictor initialized (using mock predictions)")
            
            # Try to load trained model if exists
            model_path = MODELS_DIR / "best_model.keras"
            if model_path.exists():
                predictor.load_model(str(model_path))
                LOGGER.info(f"✅ Loaded trained model from {model_path}")
        except Exception as e:
            LOGGER.warning(f"⚠️ Predictor initialization: {e}")
            predictor = None
    else:
        LOGGER.warning("⚠️ Model module not available - predictions will be simulated")
    
    LOGGER.info(f"📂 Web interface: {WEB_DIR}")
    LOGGER.info(f"📂 Static files: {STATIC_DIR}")
    LOGGER.info("=" * 60)


# ============================================================================
# WEB INTERFACE ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard web interface."""
    index_path = TEMPLATES_DIR / "index.html"
    
    if index_path.exists():
        with open(index_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        # Return a basic HTML response if template not found
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Kanyakumari Ocean Prediction</title>
            <style>
                body { font-family: Arial; padding: 40px; background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; min-height: 100vh; }
                h1 { font-size: 2.5em; }
                a { color: #00d4ff; }
            </style>
        </head>
        <body>
            <h1>🌊 Kanyakumari Ocean Wave Prediction System</h1>
            <p>Web interface template not found. API is running.</p>
            <p>Access API documentation: <a href="/docs">/docs</a></p>
            <p>Health check: <a href="/health">/health</a></p>
        </body>
        </html>
        """)


# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Kanyakumari Ocean Wave Prediction System",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "ocean_monitor": monitor is not None,
            "wave_predictor": predictor is not None,
            "model_loaded": predictor is not None and predictor.model is not None,
            "web_interface": (TEMPLATES_DIR / "index.html").exists(),
        },
        "location": {
            "name": "Kanyakumari",
            "latitude": KANYAKUMARI_LAT,
            "longitude": KANYAKUMARI_LON,
        }
    }


@app.get("/api/status")
async def api_status():
    """Detailed API status information."""
    return {
        "api_name": "Kanyakumari Ocean Wave & Tsunami Prediction API",
        "version": "2.0.0",
        "status": "operational",
        "monitored_region": KANYAKUMARI_REGION,
        "monitoring_points": MONITORING_POINTS,
        "data_sources": {
            "marine_data": "Open-Meteo Marine API",
            "weather_data": "Open-Meteo Weather API",
            "earthquake_data": "USGS Earthquake Catalog",
            "tsunami_bulletins": ["PTWC RSS", "NTWC RSS"],
        },
        "model_info": {
            "architecture": "Multimodal CNN-LSTM Hybrid",
            "wave_classes": ["NORMAL", "MODERATE", "HIGH", "EXTREME"],
            "tsunami_classes": ["NONE", "LOW", "HIGH"],
        },
        "endpoints": {
            "dashboard": "/",
            "health": "/health",
            "current_conditions": "/api/current",
            "full_prediction": "/api/predict",
            "marine_data": "/api/marine",
            "weather_data": "/api/weather",
            "earthquakes": "/api/earthquakes",
            "bulletins": "/api/bulletins",
            "historical": "/api/historical",
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.get("/api/predict")
async def get_full_prediction(
    latitude: float = Query(KANYAKUMARI_LAT, description="Latitude (default: Kanyakumari)"),
    longitude: float = Query(KANYAKUMARI_LON, description="Longitude (default: Kanyakumari)"),
    include_forecast: bool = Query(True, description="Include wave forecast data"),
    forecast_hours: int = Query(48, ge=12, le=168, description="Forecast hours (12-168)")
):
    """Get comprehensive ocean prediction with all data sources.
    
    This endpoint combines:
    - Real-time marine conditions and forecasts
    - Weather conditions
    - Recent earthquake activity with tsunami risk
    - Official tsunami bulletins
    - Model-based wave and tsunami predictions
    """
    try:
        # Create monitor for requested location
        location_name = "Kanyakumari" if (latitude == KANYAKUMARI_LAT and longitude == KANYAKUMARI_LON) else "Custom Location"
        loc_monitor = KanyakumariOceanMonitor(latitude, longitude, location_name)
        
        # Fetch all data
        all_data = loc_monitor.fetch_all_data()
        
        # Prepare model input and get prediction
        prediction_result = await _get_model_prediction(loc_monitor, latitude, longitude)
        
        # Build response
        response = {
            "success": True,
            "location": {
                "name": location_name,
                "latitude": latitude,
                "longitude": longitude,
                "region": KANYAKUMARI_REGION["name"],
            },
            "prediction": prediction_result,
            "current_conditions": {
                "marine": all_data.get("marine", {}).get("current", {}),
                "weather": {
                    "temperature_c": all_data.get("weather", {}).get("temperature_c"),
                    "humidity_percent": all_data.get("weather", {}).get("humidity_percent"),
                    "wind_speed_kmh": all_data.get("weather", {}).get("wind_speed_kmh"),
                    "wind_direction": all_data.get("weather", {}).get("wind_direction_name"),
                    "conditions": all_data.get("weather", {}).get("weather_description"),
                },
            },
            "wave_statistics": all_data.get("marine", {}).get("statistics", {}),
            "max_wave_forecast": all_data.get("marine", {}).get("max_wave_forecast", {}),
            "seismic_activity": {
                "earthquake_count": all_data.get("earthquakes", {}).get("total_count", 0),
                "tsunami_risk_count": all_data.get("earthquakes", {}).get("tsunami_risk_count", 0),
                "highest_magnitude": all_data.get("earthquakes", {}).get("highest_magnitude"),
                "tsunami_risk_events": all_data.get("earthquakes", {}).get("tsunami_risk_events", [])[:5],
            },
            "active_warnings": all_data.get("bulletins", {}).get("active_warnings", []),
            "overall_risk": all_data.get("overall_risk_assessment", {}),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Include forecast data if requested
        if include_forecast:
            forecast_data = all_data.get("marine", {}).get("forecast_data", [])
            response["forecast"] = forecast_data[:forecast_hours] if forecast_data else []
        
        return response
        
    except Exception as e:
        LOGGER.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/current")
async def get_current_conditions():
    """Get current ocean and weather conditions for Kanyakumari.
    
    Quick endpoint for dashboard updates showing:
    - Current wave height, period, and direction
    - Weather conditions (temperature, wind, humidity)
    - Wave severity classification
    - Quick risk assessment
    """
    try:
        if monitor is None:
            raise HTTPException(status_code=503, detail="Ocean monitor not initialized")
        
        marine = monitor.fetch_marine_data(forecast_hours=24)
        weather = monitor.fetch_weather_data()
        
        current = marine.get("current", {})
        
        # Quick prediction
        prediction = await _get_model_prediction(monitor, KANYAKUMARI_LAT, KANYAKUMARI_LON)
        
        return {
            "success": True,
            "location": {
                "name": "Kanyakumari",
                "latitude": KANYAKUMARI_LAT,
                "longitude": KANYAKUMARI_LON,
            },
            "wave_conditions": {
                "height_m": current.get("wave_height_m"),
                "period_s": current.get("wave_period_s"),
                "direction": current.get("wave_direction_name"),
                "condition": current.get("wave_condition_display"),
                "swell_height_m": current.get("swell_height_m"),
            },
            "weather": {
                "temperature_c": weather.get("temperature_c"),
                "feels_like_c": weather.get("feels_like_c"),
                "humidity_percent": weather.get("humidity_percent"),
                "wind_speed_kmh": weather.get("wind_speed_kmh"),
                "wind_direction": weather.get("wind_direction_name"),
                "conditions": weather.get("weather_description"),
                "pressure_hpa": weather.get("pressure_hpa"),
            },
            "prediction": prediction,
            "max_wave_next_24h": marine.get("max_wave_forecast", {}),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Error fetching current conditions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _get_model_prediction(
    loc_monitor: KanyakumariOceanMonitor,
    latitude: float,
    longitude: float
) -> Dict[str, Any]:
    """Get model prediction for location."""
    
    if predictor is None:
        # Return mock prediction
        return _generate_mock_prediction(loc_monitor)
    
    try:
        # Prepare model input
        sequence, meta = loc_monitor.prepare_model_input(hours=24)
        
        # Create dummy image input (can be replaced with actual satellite data)
        dummy_image = np.random.randn(128, 128, 3).astype(np.float32)
        
        # Get prediction
        result = predictor.predict(dummy_image, sequence)
        
        return {
            "wave_severity": result.get("wave_class", "NORMAL"),
            "tsunami_risk": result.get("tsunami_class", "NONE"),
            "hazard_index": result.get("hazard_probability_index", 0.0),
            "predicted_wave_height_m": result.get("predicted_wave_height_m"),
            "wave_probabilities": result.get("wave_probabilities", {}),
            "tsunami_probabilities": result.get("tsunami_probabilities", {}),
            "confidence": max(result.get("wave_probabilities", {}).values()) if result.get("wave_probabilities") else 0.8,
            "is_simulated": result.get("is_mock", False),
        }
        
    except Exception as e:
        LOGGER.warning(f"Model prediction failed: {e}. Using simulation.")
        return _generate_mock_prediction(loc_monitor)


def _generate_mock_prediction(loc_monitor: KanyakumariOceanMonitor) -> Dict[str, Any]:
    """Generate simulated prediction based on current conditions."""
    
    marine_data = loc_monitor.fetch_marine_data(forecast_hours=24)
    current = marine_data.get("current", {})
    wave_height = current.get("wave_height_m", 0) or 0
    
    # Determine wave severity based on actual wave height
    if wave_height < 1.0:
        wave_severity = "NORMAL"
        wave_probs = {"NORMAL": 0.85, "MODERATE": 0.12, "HIGH": 0.025, "EXTREME": 0.005}
    elif wave_height < 2.0:
        wave_severity = "MODERATE"
        wave_probs = {"NORMAL": 0.25, "MODERATE": 0.60, "HIGH": 0.12, "EXTREME": 0.03}
    elif wave_height < 4.0:
        wave_severity = "HIGH"
        wave_probs = {"NORMAL": 0.05, "MODERATE": 0.20, "HIGH": 0.65, "EXTREME": 0.10}
    else:
        wave_severity = "EXTREME"
        wave_probs = {"NORMAL": 0.02, "MODERATE": 0.08, "HIGH": 0.25, "EXTREME": 0.65}
    
    # Tsunami risk (typically low unless there's seismic activity)
    tsunami_risk = "NONE"
    tsunami_probs = {"NONE": 0.92, "LOW": 0.06, "HIGH": 0.02}
    
    # Calculate hazard index
    hazard_index = (
        0.4 * (wave_probs["MODERATE"] * 0.3 + wave_probs["HIGH"] * 0.7 + wave_probs["EXTREME"] * 1.0) +
        0.6 * (tsunami_probs["LOW"] * 0.5 + tsunami_probs["HIGH"] * 1.0)
    )
    
    return {
        "wave_severity": wave_severity,
        "tsunami_risk": tsunami_risk,
        "hazard_index": round(hazard_index, 3),
        "predicted_wave_height_m": round(wave_height * 1.1, 2),  # Slight increase for peak prediction
        "wave_probabilities": wave_probs,
        "tsunami_probabilities": tsunami_probs,
        "confidence": max(wave_probs.values()),
        "is_simulated": True,
        "note": "Prediction based on current conditions analysis",
    }


# ============================================================================
# DATA ENDPOINTS
# ============================================================================

@app.get("/api/marine")
async def get_marine_data(
    latitude: float = Query(KANYAKUMARI_LAT),
    longitude: float = Query(KANYAKUMARI_LON),
    forecast_hours: int = Query(72, ge=12, le=168)
):
    """Get marine wave data and forecast."""
    try:
        location_name = "Kanyakumari" if (latitude == KANYAKUMARI_LAT and longitude == KANYAKUMARI_LON) else "Custom"
        loc_monitor = KanyakumariOceanMonitor(latitude, longitude, location_name)
        return loc_monitor.fetch_marine_data(forecast_hours=forecast_hours)
    except Exception as e:
        LOGGER.error(f"Error fetching marine data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/weather")
async def get_weather_data(
    latitude: float = Query(KANYAKUMARI_LAT),
    longitude: float = Query(KANYAKUMARI_LON)
):
    """Get current weather conditions."""
    try:
        location_name = "Kanyakumari" if (latitude == KANYAKUMARI_LAT and longitude == KANYAKUMARI_LON) else "Custom"
        loc_monitor = KanyakumariOceanMonitor(latitude, longitude, location_name)
        return loc_monitor.fetch_weather_data()
    except Exception as e:
        LOGGER.error(f"Error fetching weather data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/earthquakes")
async def get_earthquake_data(
    min_magnitude: float = Query(5.0, ge=3.0, le=9.0, description="Minimum earthquake magnitude"),
    hours_back: int = Query(168, ge=24, le=720, description="Hours of history to search"),
    max_distance_km: float = Query(5000, ge=500, le=10000, description="Maximum distance from Kanyakumari")
):
    """Get recent earthquake data with tsunami risk assessment."""
    try:
        if monitor is None:
            raise HTTPException(status_code=503, detail="Monitor not initialized")
        return monitor.fetch_earthquakes(
            min_magnitude=min_magnitude,
            hours_back=hours_back,
            max_distance_km=max_distance_km
        )
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Error fetching earthquake data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bulletins")
async def get_tsunami_bulletins():
    """Get official tsunami bulletins from warning centers."""
    try:
        if monitor is None:
            raise HTTPException(status_code=503, detail="Monitor not initialized")
        return monitor.fetch_tsunami_bulletins()
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Error fetching bulletins: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/historical")
async def get_historical_data():
    """Get historical tsunami data for Kanyakumari region."""
    try:
        if monitor is None:
            raise HTTPException(status_code=503, detail="Monitor not initialized")
        return monitor.get_historical_tsunamis()
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Error fetching historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/all")
async def get_all_data():
    """Get all available data from all sources."""
    try:
        if monitor is None:
            raise HTTPException(status_code=503, detail="Monitor not initialized")
        return monitor.fetch_all_data()
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Error fetching all data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MONITORING POINTS ENDPOINT
# ============================================================================

@app.get("/api/monitoring-points")
async def get_monitoring_points():
    """Get list of available monitoring points."""
    return {
        "success": True,
        "primary_location": {
            "name": "Kanyakumari",
            "latitude": KANYAKUMARI_LAT,
            "longitude": KANYAKUMARI_LON,
            "description": "Southernmost tip of India - confluence of three seas",
        },
        "monitoring_points": MONITORING_POINTS,
        "region": KANYAKUMARI_REGION,
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Run the API server."""
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    print("\n" + "=" * 60)
    print("🌊 KANYAKUMARI OCEAN WAVE & TSUNAMI PREDICTION SYSTEM")
    print("=" * 60)
    print(f"📍 Location: Kanyakumari ({KANYAKUMARI_LAT}, {KANYAKUMARI_LON})")
    print(f"🌐 Server: http://{host}:{port}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "kanyakumari_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
