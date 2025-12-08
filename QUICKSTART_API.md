# Quick Start Guide - Real-Time Monitoring API

This guide will help you get the real-time ocean wave disaster prediction API up and running.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

## Starting the API Server

### Option 1: Basic Server (Development)

```bash
cd src
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- Base URL: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

### Option 2: Production Server

```bash
cd src
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

## Quick Test

Once the server is running, test it:

```bash
# Check health
curl http://localhost:8000/health

# Get prediction for a location (Monterey Bay, CA)
curl "http://localhost:8000/predict?latitude=36.78&longitude=-122.40"

# Get tsunami bulletins
curl "http://localhost:8000/bulletins?active_only=true"

# Get recent earthquakes
curl "http://localhost:8000/earthquakes?min_magnitude=5.0&tsunami_risk_only=true"
```

## Using the Interactive Documentation

The easiest way to explore the API is through the interactive Swagger UI:

1. Open http://localhost:8000/docs in your browser
2. Click on any endpoint to expand it
3. Click "Try it out"
4. Fill in the parameters
5. Click "Execute"
6. View the response

## Example Responses

### `/predict` - Live Prediction

Request:
```bash
curl "http://localhost:8000/predict?latitude=36.78&longitude=-122.40"
```

Response:
```json
{
  "predicted_class": "NORMAL",
  "predicted_class_index": 0,
  "hazard_probability_index": 0.0925,
  "probabilities": {
    "NORMAL": 0.85,
    "MODERATE": 0.12,
    "GIANT": 0.03
  },
  "confidence": 0.85,
  "location": {
    "latitude": 36.78,
    "longitude": -122.4
  },
  "data_sources": {
    "open_meteo": true,
    "ndbc": true
  },
  "timestamp": "2025-12-08T08:00:00.000000"
}
```

### `/bulletins` - Tsunami Warnings

Request:
```bash
curl "http://localhost:8000/bulletins"
```

Response:
```json
{
  "count": 2,
  "bulletins": [
    {
      "title": "Tsunami Information Statement",
      "summary": "There is no tsunami threat from this earthquake...",
      "link": "https://www.tsunami.gov/...",
      "published": "2025-12-08T07:30:00",
      "source": "PTWC",
      "severity": "INFORMATION"
    }
  ],
  "timestamp": "2025-12-08T08:00:00.000000"
}
```

### `/earthquakes` - Recent Seismic Events

Request:
```bash
curl "http://localhost:8000/earthquakes?min_magnitude=6.0&tsunami_risk_only=true"
```

Response:
```json
{
  "count": 3,
  "events": [
    {
      "time": "2025-12-07T15:23:45.000000",
      "latitude": 38.12,
      "longitude": 142.34,
      "depth_km": 35.2,
      "magnitude": 6.5,
      "magnitude_type": "mww",
      "place": "150 km E of Sendai, Japan",
      "tsunami_flag": 1,
      "url": "https://earthquake.usgs.gov/..."
    }
  ],
  "timestamp": "2025-12-08T08:00:00.000000"
}
```

## Python Client Example

```python
import httpx

# Get prediction
response = httpx.get(
    "http://localhost:8000/predict",
    params={"latitude": 36.78, "longitude": -122.40}
)
data = response.json()

print(f"Risk Level: {data['predicted_class']}")
print(f"Hazard Index: {data['hazard_probability_index']:.3f}")

# Check for active warnings
response = httpx.get(
    "http://localhost:8000/bulletins",
    params={"active_only": True}
)
bulletins = response.json()

if bulletins['count'] > 0:
    print(f"⚠️  {bulletins['count']} active tsunami warning(s)!")
```

## Running Example Scripts

```bash
# Test all data fetchers
python examples/test_realtime.py

# Test API endpoints
python examples/test_api.py

# Complete usage example (requires API to be running)
python examples/api_usage_example.py
```

## Configuration

### Using a Trained Model

By default, the API uses mock predictions. To use a trained model:

```bash
curl "http://localhost:8000/predict?latitude=36.78&longitude=-122.40&model_path=/path/to/best_model.h5&scaler_path=/path/to/scaler.pkl"
```

Or set environment variables:

```bash
export MODEL_PATH=/path/to/best_model.h5
export SCALER_PATH=/path/to/scaler.pkl
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### NDBC Station Configuration

To use a specific NDBC buoy station:

```bash
curl "http://localhost:8000/marine-data?latitude=36.78&longitude=-122.40&source=ndbc&station_id=46042"
```

## Troubleshooting

### API won't start
- Check if port 8000 is already in use: `lsof -i :8000`
- Try a different port: `uvicorn api:app --port 8080`

### Network errors
- The data fetchers require internet access to external APIs
- Check your firewall settings
- Verify the external APIs are accessible:
  - Open-Meteo: https://marine-api.open-meteo.com/
  - USGS: https://earthquake.usgs.gov/
  - NOAA NDBC: https://www.ndbc.noaa.gov/

### Import errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Run from the correct directory (project root or src/)

## Next Steps

- Deploy to production with proper authentication
- Add caching for external API calls
- Integrate with your trained model
- Set up monitoring and alerting
- Configure CORS for web clients

For more details, see the main README.md file.
