"""FastAPI service for real-time ocean wave disaster prediction monitoring.

Provides REST API endpoints for:
- Live predictions using real-time data
- Tsunami bulletin aggregation
- Health monitoring
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")

try:
    import tensorflow as tf
except ImportError:
    tf = None

from realtime.open_meteo import fetch_open_meteo_marine, prepare_wave_features
from realtime.ndbc import fetch_ndbc_latest, prepare_ndbc_features, find_nearest_station
from realtime.usgs_earthquake import (
    fetch_usgs_earthquakes,
    filter_tsunami_risk_events,
    calculate_tsunami_arrival_estimate,
)
from realtime.tsunami_bulletins import fetch_tsunami_bulletins, get_active_warnings
from realtime.data_utils import (
    merge_data_sources,
    build_model_input,
    format_prediction_result,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ocean Wave Disaster Prediction API",
    description="Real-time monitoring and prediction service for ocean wave disasters",
    version="1.0.0",
)

# Global model instance (loaded on startup)
MODEL = None
SCALER_PATH = None
MODEL_CONFIG = {
    "image_size": (128, 128),
    "seq_len": 12,
    "seq_features": 5,
    "feature_columns": ["Hs", "Hmax", "SST", "WindSpeed", "PeakWaveDirection"],
    "class_names": ["NORMAL", "MODERATE", "GIANT"],
}


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup (if available)."""
    global MODEL, SCALER_PATH
    LOGGER.info("Starting Ocean Wave Disaster Prediction API")
    # Model loading will be handled by the /predict endpoint
    # This allows the API to run even without a trained model


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Ocean Wave Disaster Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "bulletins": "/bulletins",
            "earthquakes": "/earthquakes",
            "marine_data": "/marine-data",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": MODEL is not None,
        "tensorflow_available": tf is not None,
    }


@app.get("/predict")
async def predict(
    latitude: float = Query(..., description="Latitude in decimal degrees"),
    longitude: float = Query(..., description="Longitude in decimal degrees"),
    use_ndbc: bool = Query(True, description="Attempt to use NDBC buoy data"),
    model_path: Optional[str] = Query(None, description="Path to model file"),
    scaler_path: Optional[str] = Query(None, description="Path to scaler file"),
):
    """Generate live prediction for a location using real-time data.
    
    This endpoint:
    1. Fetches real-time marine data (Open-Meteo, optionally NDBC)
    2. Prepares model input tensors
    3. Runs inference (mock if no model provided)
    4. Returns prediction with probabilities and hazard index
    """
    try:
        # Fetch real-time data
        LOGGER.info(f"Fetching data for location: ({latitude}, {longitude})")
        
        # Get Open-Meteo marine forecast
        meteo_df = fetch_open_meteo_marine(latitude, longitude, hours=48)
        meteo_features = prepare_wave_features(meteo_df, MODEL_CONFIG["feature_columns"])
        
        # Try to get NDBC data if requested
        ndbc_features = None
        if use_ndbc:
            station = find_nearest_station(latitude, longitude, max_distance_km=500)
            if station:
                LOGGER.info(f"Using NDBC station: {station}")
                ndbc_df = fetch_ndbc_latest(station, data_type="txt")
                if not ndbc_df.empty:
                    ndbc_features = prepare_ndbc_features(
                        ndbc_df, 
                        data_type="txt",
                        target_features=MODEL_CONFIG["feature_columns"]
                    )
        
        # Merge data sources
        merged_data = merge_data_sources(
            open_meteo_df=meteo_features,
            ndbc_df=ndbc_features,
            feature_columns=MODEL_CONFIG["feature_columns"],
        )
        
        if merged_data.empty:
            raise HTTPException(
                status_code=503,
                detail="Failed to fetch real-time data from any source"
            )
        
        # Build model input
        img_tensor, seq_tensor = build_model_input(
            sequence_df=merged_data,
            feature_columns=MODEL_CONFIG["feature_columns"],
            seq_len=MODEL_CONFIG["seq_len"],
            image_size=MODEL_CONFIG["image_size"],
            scaler_path=scaler_path,
        )
        
        # Run prediction
        if model_path and tf is not None:
            # Load and use actual model
            try:
                global MODEL
                if MODEL is None:
                    LOGGER.info(f"Loading model from {model_path}")
                    MODEL = tf.keras.models.load_model(model_path)
                
                predictions = MODEL.predict([img_tensor, seq_tensor], verbose=0)
                probabilities = predictions[0]
            except Exception as exc:
                LOGGER.warning(f"Model prediction failed: {exc}. Using mock prediction.")
                probabilities = _mock_prediction(merged_data)
        else:
            # Mock prediction
            LOGGER.info("No model provided, using mock prediction")
            probabilities = _mock_prediction(merged_data)
        
        # Format result
        result = format_prediction_result(probabilities, MODEL_CONFIG["class_names"])
        result["location"] = {
            "latitude": latitude,
            "longitude": longitude,
        }
        result["data_sources"] = {
            "open_meteo": not meteo_features.empty,
            "ndbc": ndbc_features is not None and not ndbc_features.empty,
        }
        result["timestamp"] = datetime.utcnow().isoformat()
        
        return result
        
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error(f"Prediction failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/bulletins")
async def get_bulletins(
    sources: Optional[str] = Query(None, description="Comma-separated list: ptwc,ntwc"),
    active_only: bool = Query(False, description="Return only active warnings"),
):
    """Fetch tsunami bulletins from NOAA warning centers.
    
    Returns authoritative tsunami warnings, advisories, and information messages.
    """
    try:
        source_list = None
        if sources:
            source_list = [s.strip().lower() for s in sources.split(",")]
        
        bulletins = fetch_tsunami_bulletins(sources=source_list)
        
        if active_only:
            bulletins = get_active_warnings(bulletins)
        
        return {
            "count": len(bulletins),
            "bulletins": [
                {
                    "title": b["title"],
                    "summary": b["summary"],
                    "link": b["link"],
                    "published": b["published"].isoformat(),
                    "source": b["source"],
                    "severity": b["severity"],
                }
                for b in bulletins
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as exc:
        LOGGER.error(f"Failed to fetch bulletins: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/earthquakes")
async def get_earthquakes(
    min_magnitude: float = Query(5.0, description="Minimum earthquake magnitude"),
    hours_back: int = Query(24, description="Hours of history to fetch"),
    tsunami_risk_only: bool = Query(False, description="Filter for tsunami-capable events"),
    target_lat: Optional[float] = Query(None, description="Target latitude for ETA calculation"),
    target_lon: Optional[float] = Query(None, description="Target longitude for ETA calculation"),
):
    """Fetch recent earthquakes from USGS catalog.
    
    Optionally filter for tsunami-capable events and calculate arrival estimates.
    """
    try:
        earthquakes = fetch_usgs_earthquakes(
            min_magnitude=min_magnitude,
            hours_back=hours_back,
        )
        
        if earthquakes.empty:
            return {
                "count": 0,
                "events": [],
                "timestamp": datetime.utcnow().isoformat(),
            }
        
        if tsunami_risk_only:
            earthquakes = filter_tsunami_risk_events(earthquakes)
        
        # Convert to list of dicts
        events = []
        for idx, row in earthquakes.iterrows():
            event = {
                "time": idx.isoformat(),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "depth_km": float(row["depth"]),
                "magnitude": float(row["magnitude"]),
                "magnitude_type": row["magnitude_type"],
                "place": row["place"],
                "tsunami_flag": int(row["tsunami"]),
                "url": row["url"],
            }
            
            # Calculate ETA if target location provided
            if target_lat is not None and target_lon is not None:
                eta = calculate_tsunami_arrival_estimate(
                    event_lat=row["latitude"],
                    event_lon=row["longitude"],
                    target_lat=target_lat,
                    target_lon=target_lon,
                )
                event["tsunami_eta"] = eta
            
            events.append(event)
        
        return {
            "count": len(events),
            "events": events,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
    except Exception as exc:
        LOGGER.error(f"Failed to fetch earthquakes: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/marine-data")
async def get_marine_data(
    latitude: float = Query(..., description="Latitude in decimal degrees"),
    longitude: float = Query(..., description="Longitude in decimal degrees"),
    source: str = Query("open-meteo", description="Data source: open-meteo or ndbc"),
    station_id: Optional[str] = Query(None, description="NDBC station ID (if source=ndbc)"),
):
    """Fetch raw marine observation/forecast data.
    
    Returns wave heights, periods, wind speed, etc. from specified source.
    """
    try:
        if source == "open-meteo":
            data = fetch_open_meteo_marine(latitude, longitude, hours=48)
            
            if data.empty:
                return {
                    "source": "open-meteo",
                    "location": {"latitude": latitude, "longitude": longitude},
                    "count": 0,
                    "data": [],
                    "timestamp": datetime.utcnow().isoformat(),
                }
            
            # Convert to records
            records = []
            for idx, row in data.iterrows():
                record = {"time": idx.isoformat()}
                record.update({k: float(v) if not pd.isna(v) else None for k, v in row.items()})
                records.append(record)
            
            return {
                "source": "open-meteo",
                "location": {"latitude": latitude, "longitude": longitude},
                "count": len(records),
                "data": records,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        elif source == "ndbc":
            if not station_id:
                # Try to find nearest station
                station_id = find_nearest_station(latitude, longitude, max_distance_km=500)
                if not station_id:
                    raise HTTPException(
                        status_code=404,
                        detail="No NDBC station found within 500km. Provide station_id explicitly."
                    )
            
            data = fetch_ndbc_latest(station_id, data_type="txt")
            
            if data.empty:
                return {
                    "source": "ndbc",
                    "station_id": station_id,
                    "count": 0,
                    "data": [],
                    "timestamp": datetime.utcnow().isoformat(),
                }
            
            # Convert to records
            records = []
            for idx, row in data.iterrows():
                record = {"time": idx.isoformat()}
                record.update({k: float(v) if not pd.isna(v) else None for k, v in row.items()})
                records.append(record)
            
            return {
                "source": "ndbc",
                "station_id": station_id,
                "count": len(records),
                "data": records,
                "timestamp": datetime.utcnow().isoformat(),
            }
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown source: {source}. Use 'open-meteo' or 'ndbc'"
            )
            
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error(f"Failed to fetch marine data: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


def _mock_prediction(data: "pd.DataFrame") -> np.ndarray:
    """Generate mock prediction based on data characteristics.
    
    This is used when no trained model is available.
    """
    # Simple heuristic based on wave height
    if "Hs" in data.columns and not data["Hs"].isna().all():
        mean_hs = data["Hs"].mean()
        
        # Simple classification based on significant wave height
        if mean_hs < 2.0:
            # NORMAL
            probs = np.array([0.85, 0.12, 0.03])
        elif mean_hs < 4.0:
            # MODERATE
            probs = np.array([0.20, 0.70, 0.10])
        else:
            # GIANT
            probs = np.array([0.05, 0.25, 0.70])
    else:
        # No data available, default to NORMAL with low confidence
        probs = np.array([0.60, 0.30, 0.10])
    
    return probs


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
