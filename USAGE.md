# Usage Guide: Ocean Wave Disaster Prediction System

This guide provides detailed instructions for using the Ocean Wave Disaster Prediction System's web interface.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Using the Web Interface](#using-the-web-interface)
3. [Understanding the Results](#understanding-the-results)
4. [Data Sources](#data-sources)
5. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Internet connection (for fetching real-time data)
- Modern web browser (Chrome, Firefox, Safari, or Edge)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/vsiva763-git/ocean-wave-disaster-prediction.git
cd ocean-wave-disaster-prediction
```

2. Start the web interface:
```bash
# Linux/Mac
./start_web.sh

# Windows
start_web.bat
```

3. Open your browser to `http://localhost:8000`

## Using the Web Interface

### Method 1: Interactive Map Selection

1. **Locate the Interactive Map** in the lower left section of the page
2. **Click anywhere** on the map where you want to check ocean conditions
3. The system will:
   - Place a marker at the selected location
   - Automatically fetch real-time data
   - Display predictions in the right panel

### Method 2: Location Presets

For quick access to major ocean regions:

1. **Click one of the preset buttons**:
   - **Bay of Bengal**: Eastern India, Bangladesh coast
   - **Arabian Sea**: Western India, Arabian Peninsula
   - **Pacific Ocean**: Largest ocean, Americas to Asia
   - **Atlantic Ocean**: Americas to Europe/Africa
   - **Indian Ocean**: Africa to Australia

2. The map will zoom to that region and show predictions

### Method 3: Custom Coordinates

For precise location selection:

1. **Enter Latitude** (e.g., 15.0)
   - Range: -90 to 90
   - Positive = North, Negative = South

2. **Enter Longitude** (e.g., 88.0)
   - Range: -180 to 180
   - Positive = East, Negative = West

3. **Click "Get Prediction"**

## Understanding the Results

### Current Ocean Conditions & Prediction

#### Hazard Level Badge
The system classifies ocean wave risk into three levels:

- **🟢 NORMAL** (Green)
  - Safe conditions
  - Regular wave patterns
  - Low risk to coastal areas

- **🟡 MODERATE** (Yellow)
  - Elevated risk
  - Increased wave activity
  - Caution advised for small vessels

- **🔴 GIANT** (Red)
  - High risk
  - Potentially dangerous waves
  - Warning for all maritime activities

#### Hazard Probability Index (HPI)
A continuous risk score from 0.0 to 1.0:
- **0.0 - 0.3**: Low risk (safe conditions)
- **0.3 - 0.6**: Moderate risk (caution needed)
- **0.6 - 1.0**: High risk (dangerous conditions)

The HPI is calculated as: `HPI = 0.0 × P(NORMAL) + 0.5 × P(MODERATE) + 1.0 × P(GIANT)`

#### Class Probabilities
Shows the model's confidence for each hazard level:
- Three horizontal bars representing NORMAL, MODERATE, and GIANT
- Percentages sum to 100%
- Higher percentage = More confident prediction

**Example:**
```
NORMAL:   85% ████████████████████
MODERATE: 12% ███
GIANT:     3% █
```
This indicates 85% confidence in NORMAL conditions.

#### Data Sources
Shows which real-time data sources were successfully accessed:
- ✓ **Open-Meteo Marine API**: Global ocean forecasts
- ✓ **NDBC Buoy Data**: High-precision buoy measurements (when available)

### Recent Tsunami Activity & Seismic Events

#### Earthquakes Tab

**Filters:**
- **Min Magnitude**: Set minimum earthquake magnitude (default: 5.0)
- **Tsunami Risk Only**: Filter for earthquakes with tsunami potential

**Information Displayed:**
- **Magnitude**: Earthquake strength (M5.0+)
- **Location**: Geographic description
- **Depth**: Earthquake depth in kilometers
- **Tsunami Flag**: ⚠️ indicates tsunami risk
- **Estimated Arrival**: Time for potential tsunami to reach selected location (if applicable)

**Click "Refresh"** to load the latest 24 hours of earthquake data.

#### Tsunami Bulletins Tab

**Filters:**
- **Active Only**: Show only current warnings/advisories

**Information Displayed:**
- **Severity**: WARNING (red), ADVISORY (yellow), or WATCH (blue)
- **Title**: Brief description
- **Summary**: Detailed information
- **Published Time**: When the bulletin was issued
- **Source**: PTWC (Pacific) or NTWC (National)

**Click "Refresh"** to load the latest tsunami bulletins.

## Data Sources

The system integrates data from multiple authoritative sources:

### 1. Open-Meteo Marine API
- **Type**: Ocean forecasts
- **Coverage**: Global
- **Update Frequency**: Hourly
- **Data**: Wave height, period, direction, wind speed
- **Cost**: Free, no API key required

### 2. NOAA NDBC (National Data Buoy Center)
- **Type**: Real-time buoy observations
- **Coverage**: US coastal waters, Pacific Ocean
- **Update Frequency**: Real-time (varies by buoy)
- **Data**: Wave measurements, water temperature, wind
- **Cost**: Free

### 3. USGS Earthquake Catalog
- **Type**: Seismic event monitoring
- **Coverage**: Global
- **Update Frequency**: Real-time
- **Data**: Magnitude, location, depth, tsunami potential
- **Cost**: Free

### 4. NOAA Tsunami Warning Centers
- **Type**: Official tsunami warnings
- **Coverage**: Pacific (PTWC), National (NTWC)
- **Update Frequency**: Real-time
- **Data**: Warnings, advisories, watches, information messages
- **Cost**: Free

## Example Use Cases

### Use Case 1: Beach Safety Check
**Scenario**: Planning a beach trip to Goa, India

1. Click on "Arabian Sea" preset or enter coordinates (15.5, 73.8)
2. Check the Hazard Level:
   - NORMAL = Safe to visit
   - MODERATE = Exercise caution
   - GIANT = Postpone visit
3. Monitor tsunami bulletins for any active warnings

### Use Case 2: Maritime Route Planning
**Scenario**: Ship navigation from Chennai to Singapore

1. Enter waypoint coordinates manually
2. Check HPI values along the route
3. Review earthquake activity for tsunami risks
4. Avoid routes with MODERATE or GIANT predictions

### Use Case 3: Coastal Disaster Monitoring
**Scenario**: Emergency management for coastal communities

1. Select region using preset buttons
2. Monitor HPI trends (refresh periodically)
3. Check Earthquake tab for tsunami triggers
4. Monitor Tsunami Bulletins for official warnings
5. Take action if HPI > 0.6 or active tsunami warnings

## Troubleshooting

### Problem: Map Not Loading
**Solutions:**
- Check internet connection
- Disable browser ad blockers
- Try a different browser
- Verify firewall isn't blocking external map tiles

### Problem: "Failed to fetch data" Error
**Possible Causes:**
1. **External API temporarily unavailable**
   - Wait a few minutes and try again
   - System will use mock predictions if needed

2. **Network connectivity issues**
   - Check your internet connection
   - Verify firewall settings

3. **Coordinates out of range**
   - Ensure latitude: -90 to 90
   - Ensure longitude: -180 to 180

### Problem: Slow Response Times
**Solutions:**
- External API calls take 5-10 seconds (normal)
- Close unnecessary browser tabs
- Check network speed
- Consider implementing caching for production use

### Problem: No Earthquakes/Bulletins Displayed
**Explanations:**
- May be no recent events matching criteria
- Click "Refresh" to reload data
- Adjust magnitude filter (try lower values)
- Uncheck "Active Only" to see all bulletins

### Problem: Predictions Seem Inaccurate
**Important Notes:**
- Without a trained model, system uses mock predictions
- Mock predictions use simple heuristics (wave height thresholds)
- For accurate predictions, train the CNN+LSTM model first
- See main README.md for model training instructions

## Tips for Best Results

1. **Refresh Data Regularly**: Ocean conditions change; check back frequently
2. **Use Multiple Data Sources**: Compare predictions with official forecasts
3. **Consider Trends**: Look at multiple nearby locations for patterns
4. **Check Historical Data**: Review earthquake and bulletin history
5. **Enable Auto-Refresh**: Uncomment last line in `web/static/app.js` for 5-min updates

## API Integration

For programmatic access, use the REST API directly:

```python
import requests

# Get prediction
response = requests.get(
    "http://localhost:8000/predict",
    params={"latitude": 15.0, "longitude": 88.0}
)
data = response.json()
print(f"Hazard Level: {data['predicted_class']}")
print(f"HPI: {data['hazard_probability_index']:.3f}")
```

See `/docs` endpoint for complete API documentation.

## Support and Contributions

- **Issues**: Report bugs via GitHub Issues
- **Documentation**: See `README.md` and `web/README.md`
- **API Docs**: Available at `http://localhost:8000/docs`
- **Contributions**: Pull requests welcome!

## License

MIT License - See LICENSE file for details
