# 🌊 Kanyakumari Ocean Wave & Tsunami Prediction System

A **Multimodal CNN-LSTM Hybrid Deep Learning System** for real-time ocean wave prediction and tsunami disaster early warning, focused on the **Kanyakumari coastal region** of India.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vsiva763-git/ocean-wave-disaster-prediction/blob/main/notebooks/kanyakumari_ocean_prediction_colab.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📍 About Kanyakumari

**Kanyakumari** (8.0883°N, 77.5385°E) is the southernmost tip of the Indian subcontinent, where three water bodies converge:

- 🌊 **Arabian Sea** (West)
- 🌊 **Bay of Bengal** (East)
- 🌊 **Indian Ocean** (South)

This unique location makes it particularly vulnerable to ocean-related disasters, including the devastating **2004 Indian Ocean Tsunami** that claimed 802 lives in the region.

---

## 🎯 Features

### 🧠 Deep Learning Model

- **Multimodal CNN-LSTM Hybrid Architecture**
  - CNN backbone for spatial pattern recognition
  - Bidirectional LSTM with attention mechanism for temporal sequences
  - Multi-task learning for wave severity and tsunami risk prediction
- **Outputs:**
  - Wave Severity Classification: `NORMAL`, `MODERATE`, `HIGH`, `EXTREME`
  - Tsunami Risk Classification: `NONE`, `LOW`, `HIGH`
  - Wave Height Regression (meters)
  - Hazard Probability Index (0-1)

### 📡 Real-Time Data Sources

| Source                           | Data                                   | Update Frequency |
| -------------------------------- | -------------------------------------- | ---------------- |
| **Open-Meteo Marine API**        | Wave height, period, direction, swell  | Hourly           |
| **Open-Meteo Weather API**       | Temperature, wind, pressure, humidity  | Hourly           |
| **USGS Earthquake Catalog**      | Seismic events with tsunami assessment | Real-time        |
| **NOAA Tsunami Warning Centers** | Official bulletins (PTWC, NTWC)        | Real-time        |

### 🖥️ Web Dashboard

- **Real-time monitoring interface**
- Interactive map with Leaflet.js
- 48-hour wave forecast charts
- Earthquake monitoring with tsunami risk levels
- Historical tsunami data
- Mobile-responsive design

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES (Free APIs)                      │
├─────────────┬─────────────┬─────────────┬─────────────────────────┤
│ Open-Meteo  │ Open-Meteo  │    USGS     │     NOAA PTWC/NTWC     │
│   Marine    │   Weather   │ Earthquakes │   Tsunami Bulletins    │
└──────┬──────┴──────┬──────┴──────┬──────┴──────────┬─────────────┘
       │             │             │                  │
       └─────────────┴──────┬──────┴──────────────────┘
                            │
                   ┌────────▼────────┐
                   │  Data Fetcher   │
                   │  (kanyakumari_  │
                   │   monitor.py)   │
                   └────────┬────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
     ┌────────▼────┐  ┌─────▼─────┐  ┌────▼────────┐
     │  CNN        │  │   LSTM    │  │  Risk       │
     │  Backbone   │  │  Backbone │  │  Assessment │
     │ (Spatial)   │  │ (Temporal)│  │             │
     └─────┬───────┘  └─────┬─────┘  └──────┬──────┘
           │                │               │
           └────────┬───────┘               │
                    │                       │
           ┌────────▼────────┐              │
           │  Multimodal     │              │
           │  Fusion Layer   │◄─────────────┘
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │   Prediction    │
           │     Heads       │
           ├─────────────────┤
           │ • Wave Severity │
           │ • Tsunami Risk  │
           │ • Wave Height   │
           │ • Hazard Index  │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │   FastAPI       │
           │   REST API      │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │  Web Dashboard  │
           │  (HTML/JS/CSS)  │
           └─────────────────┘
```

---

## 📂 Project Structure

```
ocean-wave-disaster-prediction/
├── src/
│   ├── models/
│   │   └── hybrid_cnn_lstm.py    # Multimodal CNN-LSTM architecture
│   ├── kanyakumari_monitor.py    # Real-time data fetcher
│   ├── kanyakumari_api.py        # FastAPI REST API
│   ├── config.py                 # Configuration settings
│   ├── train.py                  # Model training script
│   └── inference.py              # Inference utilities
├── web/
│   ├── templates/
│   │   └── index.html            # Dashboard HTML
│   └── static/
│       ├── styles.css            # Dashboard styles
│       └── app.js                # Dashboard JavaScript
├── models/                       # Saved model weights
├── data/                         # Training data
├── notebooks/                    # Jupyter notebooks
├── examples/                     # Usage examples
├── requirements.txt              # Python dependencies
├── start_web.bat                 # Windows startup script
├── start_web.sh                  # Linux/Mac startup script
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Internet connection (for API data)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/ocean-wave-disaster-prediction.git
   cd ocean-wave-disaster-prediction
   ```

2. **Create virtual environment (recommended):**

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the System

**Option 1: Using startup script (Recommended)**

```bash
# Windows
start_web.bat

# Linux/Mac
chmod +x start_web.sh
./start_web.sh
```

**Option 2: Manual startup**

```bash
cd src
python -m uvicorn kanyakumari_api:app --host 0.0.0.0 --port 8000 --reload
```

**Access the dashboard:**

- 🌐 **Web Interface:** http://localhost:8000
- 📚 **API Documentation:** http://localhost:8000/docs
- ❤️ **Health Check:** http://localhost:8000/health

---

## 📊 API Endpoints

| Endpoint           | Method | Description                   |
| ------------------ | ------ | ----------------------------- |
| `/`                | GET    | Web dashboard interface       |
| `/health`          | GET    | System health check           |
| `/api/predict`     | GET    | Full prediction with all data |
| `/api/current`     | GET    | Current conditions only       |
| `/api/marine`      | GET    | Marine wave data              |
| `/api/weather`     | GET    | Weather conditions            |
| `/api/earthquakes` | GET    | Recent seismic activity       |
| `/api/bulletins`   | GET    | Tsunami bulletins             |
| `/api/historical`  | GET    | Historical tsunami data       |
| `/api/all`         | GET    | All data from all sources     |

### Example API Usage

```python
import httpx

# Get current conditions
response = httpx.get("http://localhost:8000/api/current")
data = response.json()

print(f"Wave Height: {data['wave_conditions']['height_m']} m")
print(f"Wave Severity: {data['prediction']['wave_severity']}")
print(f"Tsunami Risk: {data['prediction']['tsunami_risk']}")
print(f"Hazard Index: {data['prediction']['hazard_index']}")
```

---

## 🧠 Model Architecture Details

### CNN Backbone (Spatial Features)

- Input: 128x128x3 images (satellite/radar data or heatmaps)
- Layers: 4 convolutional blocks with batch normalization
- Output: 256-dimensional feature vector

### LSTM Backbone (Temporal Features)

- Input: 24 time steps × 8 features
- Layers: 2-layer Bidirectional LSTM (128, 64 units)
- Attention mechanism for temporal importance weighting
- Output: 128-dimensional feature vector

### Fusion & Output

- Multimodal fusion: Concatenation + Dense layers
- Output heads:
  - Wave Severity: 4-class softmax
  - Tsunami Risk: 3-class softmax
  - Wave Height: Linear regression

### Loss Function

```
Total Loss = 1.0 × Wave_CE + 1.5 × Tsunami_CE + 0.5 × Wave_MSE
```

(Tsunami risk weighted higher due to critical importance)

---

## 📈 Data Parameters

### Ocean Parameters (8 features)

| Parameter             | Unit    | Description             |
| --------------------- | ------- | ----------------------- |
| `wave_height`         | meters  | Significant wave height |
| `wave_period`         | seconds | Wave period             |
| `wave_direction`      | degrees | Wave direction          |
| `swell_height`        | meters  | Swell wave height       |
| `swell_period`        | seconds | Swell period            |
| `wind_wave_height`    | meters  | Wind-generated waves    |
| `wind_wave_period`    | seconds | Wind wave period        |
| `wind_wave_direction` | degrees | Wind wave direction     |

### Weather Parameters

| Parameter     | Unit |
| ------------- | ---- |
| Temperature   | °C   |
| Humidity      | %    |
| Wind Speed    | km/h |
| Pressure      | hPa  |
| Precipitation | mm   |

---

## ⚠️ Risk Classification

### Wave Severity Levels

| Level    | Wave Height | Color     |
| -------- | ----------- | --------- |
| NORMAL   | < 1.0 m     | 🟢 Green  |
| MODERATE | 1.0 - 2.5 m | 🟡 Yellow |
| HIGH     | 2.5 - 4.0 m | 🟠 Orange |
| EXTREME  | > 4.0 m     | 🔴 Red    |

### Tsunami Risk Assessment

Based on earthquake parameters:

- **Magnitude** ≥ 6.5 for tsunami potential
- **Depth** < 70 km (shallow earthquakes)
- **Distance** from Kanyakumari
- **USGS tsunami flag**

---

## 🔮 Future Enhancements

- [ ] Satellite imagery integration (Sentinel-2, Landsat)
- [ ] INCOIS (Indian Ocean) buoy data integration
- [ ] SMS/Email alert system
- [ ] Mobile app (React Native)
- [ ] Historical data training on real tsunami events
- [ ] Integration with local emergency services

---

## 📚 References

1. **2004 Indian Ocean Tsunami** - USGS Report
2. **Open-Meteo Marine API** - https://open-meteo.com/en/docs/marine-weather-api
3. **USGS Earthquake Catalog** - https://earthquake.usgs.gov/earthquakes/feed/
4. **NOAA Tsunami Warning Center** - https://www.tsunami.gov/

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

Developed for ocean disaster prediction and early warning system research.

---

<div align="center">
  <h3>🌊 Stay Safe, Stay Informed 🌊</h3>
  <p>Protecting coastal communities through AI-powered early warning</p>
</div>
