# Web Interface for Ocean Wave Disaster Prediction

## Overview
This web interface provides an interactive platform for monitoring ocean wave disasters in real-time. It combines current ocean geographical data with past tsunami data to provide predictions for selected locations.

## Features

### 1. Interactive Map
- **Global Coverage**: Click anywhere on the world map to get predictions
- **Location Presets**: Quick access to major oceans and seas:
  - Bay of Bengal
  - Arabian Sea
  - Pacific Ocean
  - Atlantic Ocean
  - Indian Ocean

### 2. Real-Time Predictions
- **Current Ocean Conditions**: Live data from Open-Meteo and NDBC buoys
- **Hazard Level Classification**: NORMAL, MODERATE, or GIANT wave risk
- **Hazard Probability Index (HPI)**: Numerical risk indicator (0-1)
- **Confidence Scores**: Probability distribution across all risk levels

### 3. Historical Data Visualization
- **Recent Earthquakes**: Monitor seismic events with tsunami potential
  - Filter by magnitude
  - Tsunami risk indicators
  - Estimated arrival times
- **Tsunami Bulletins**: Official warnings from NOAA (PTWC/NTWC)
  - Active warnings and advisories
  - Real-time RSS feed updates

## Quick Start

### 1. Start the Server

From the project root directory:

```bash
cd src
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### 2. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:8000
```

### 3. Using the Interface

#### Method 1: Click on Map
1. Click anywhere on the interactive map
2. The system automatically fetches data and displays predictions

#### Method 2: Select Preset Location
1. Click one of the preset ocean buttons (e.g., "Bay of Bengal")
2. The map zooms to that region
3. Predictions are displayed automatically

#### Method 3: Enter Custom Coordinates
1. Enter latitude and longitude manually
2. Click "Get Prediction"
3. View results in the right panel

## Data Sources

The system integrates data from multiple free APIs:

1. **Open-Meteo Marine API**: Real-time ocean forecasts
   - Wave height and direction
   - Wind speed and direction
   - Water temperature

2. **NOAA NDBC**: Buoy observations
   - Direct ocean measurements
   - High accuracy for coastal areas

3. **USGS Earthquake Catalog**: Seismic monitoring
   - Magnitude 5.0+ events
   - Tsunami potential indicators

4. **NOAA Tsunami Warning Centers**: Official bulletins
   - Pacific Tsunami Warning Center (PTWC)
   - National Tsunami Warning Center (NTWC)

## Understanding the Results

### Hazard Levels
- **NORMAL** (Green): Safe conditions, typical wave patterns
- **MODERATE** (Yellow): Elevated risk, increased wave activity
- **GIANT** (Red): High risk, potentially dangerous waves

### Hazard Probability Index (HPI)
- Range: 0.0 to 1.0
- 0.0 - 0.3: Low risk
- 0.3 - 0.6: Moderate risk
- 0.6 - 1.0: High risk

### Probability Distribution
Shows the confidence level for each hazard class:
- Higher percentage = More confident prediction
- All three bars sum to 100%

## Technical Architecture

```
User Interface (Web Browser)
    ↓
FastAPI Server (Python)
    ↓
┌─────────────────────────────────┐
│  Real-Time Data Fetchers        │
├─────────────────────────────────┤
│  • Open-Meteo Marine API        │
│  • NOAA NDBC Buoys              │
│  • USGS Earthquakes             │
│  • NOAA Tsunami Bulletins       │
└─────────────────────────────────┘
    ↓
CNN + LSTM Model
    ↓
Predictions & Visualizations
```

## File Structure

```
web/
├── templates/
│   └── index.html          # Main HTML page
└── static/
    ├── styles.css          # Styling and layout
    └── app.js              # Interactive functionality
```

## Customization

### Modify Location Presets
Edit `web/static/app.js` and update the `LOCATION_PRESETS` object:

```javascript
const LOCATION_PRESETS = {
    'my-location': { 
        lat: 12.34, 
        lon: 56.78, 
        zoom: 5, 
        name: 'My Custom Location' 
    }
};
```

### Change Styling
Edit `web/static/styles.css` to customize colors, fonts, and layout.

### Add Auto-Refresh
In `web/static/app.js`, uncomment the last line:
```javascript
startAutoRefresh(); // Refreshes predictions every 5 minutes
```

## Browser Compatibility

Tested and supported on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Mobile Support

The interface is fully responsive and works on:
- Smartphones (iOS and Android)
- Tablets
- Desktop browsers

## Troubleshooting

### Issue: Map not loading
**Solution**: Check internet connection. The map requires access to OpenStreetMap tiles.

### Issue: No predictions displayed
**Solution**: Verify the API server is running on port 8000. Check browser console for errors.

### Issue: "Failed to fetch data"
**Solution**: 
1. Ensure external APIs are accessible
2. Check firewall settings
3. Verify coordinates are within valid ranges

### Issue: Slow response times
**Solution**: 
1. External API calls may take 5-10 seconds
2. Consider implementing caching for production use
3. Check network connection speed

## Production Deployment

### Security Considerations
1. Enable HTTPS/SSL
2. Add API authentication
3. Implement rate limiting
4. Configure CORS for your domain

### Performance Optimization
1. Add Redis caching for API responses
2. Use a CDN for static files
3. Implement connection pooling
4. Enable gzip compression

### Scaling
For high-traffic deployments:
```bash
# Use gunicorn with multiple workers
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app --bind 0.0.0.0:8000
```

## API Integration

The web interface uses these API endpoints:

- `GET /predict?latitude={lat}&longitude={lon}` - Get prediction
- `GET /earthquakes?min_magnitude={mag}` - Get earthquakes
- `GET /bulletins?active_only={bool}` - Get tsunami bulletins
- `GET /health` - Check API health

See `/docs` endpoint for complete API documentation.

## License

MIT License - Same as the main project

## Support

For issues or questions:
1. Check the main README.md
2. Review API documentation at `/docs`
3. Check browser console for JavaScript errors
4. Verify API server logs for backend errors
