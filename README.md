# 🌊 Kanyakumari Ocean Wave & Tsunami Prediction System

A **Multimodal CNN-LSTM Hybrid Deep Learning System** for real-time ocean wave prediction and tsunami disaster early warning, focused on the **Kanyakumari coastal region** of India.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vsiva763-git/ocean-wave-disaster-prediction/blob/main/notebooks/run_from_src.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)

---

## 📍 About Kanyakumari

**Kanyakumari** (8.0883°N, 77.5385°E) is the southernmost tip of the Indian subcontinent, where three water bodies converge:

- 🌊 **Arabian Sea** (West)
- 🌊 **Bay of Bengal** (East)
- 🌊 **Indian Ocean** (South)

---

## 🎯 Features

### 🧠 Deep Learning Model

- **Multimodal CNN-LSTM Hybrid Architecture**
- Bidirectional LSTM with attention mechanism
- Multi-task learning for wave severity and tsunami risk prediction

### 📡 Real-Time Data Sources

| Source                      | Data                           | Update Frequency |
| --------------------------- | ------------------------------ | ---------------- |
| **Open-Meteo Marine API**   | Wave height, period, direction | Hourly           |
| **Open-Meteo Weather API**  | Temperature, wind, pressure    | Hourly           |
| **USGS Earthquake Catalog** | Seismic events                 | Real-time        |

### 🖥️ Web Dashboard

- Real-time monitoring interface
- Interactive map with Leaflet.js
- Live charts with Chart.js
- Tsunami risk alerts

---

## 🚀 Quick Start

### Option 1: Run in Google Colab (Recommended)

Click the badge above or open: [Run in Colab](https://colab.research.google.com/github/vsiva763-git/ocean-wave-disaster-prediction/blob/main/notebooks/run_from_src.ipynb)

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/vsiva763-git/ocean-wave-disaster-prediction.git
cd ocean-wave-disaster-prediction

# Install dependencies
pip install -r requirements.txt

# Start the web server
python -m uvicorn src.kanyakumari_api:app --reload

# Open http://localhost:8000 in your browser
```

---

## 📁 Project Structure

```
ocean-wave-disaster-prediction/
├── src/
│   ├── kanyakumari_api.py      # FastAPI web server
│   ├── kanyakumari_monitor.py  # Real-time data fetcher
│   ├── kanyakumari_data.py     # Data processing
│   └── models/
│       └── hybrid_cnn_lstm.py  # CNN-LSTM model
├── web/
│   ├── templates/
│   │   └── index.html          # Dashboard UI
│   └── static/
│       ├── app.js              # Frontend JavaScript
│       └── styles.css          # Styling
├── notebooks/
│   └── run_from_src.ipynb      # Colab notebook
├── models/                      # Saved trained models
├── requirements.txt
└── README.md
```

---

## 🔗 API Endpoints

| Endpoint               | Description        |
| ---------------------- | ------------------ |
| `GET /`                | Web Dashboard      |
| `GET /api/status`      | System status      |
| `GET /api/marine-data` | Current wave data  |
| `GET /api/weather`     | Weather conditions |
| `GET /api/prediction`  | AI predictions     |
| `GET /api/earthquakes` | Recent earthquakes |

---

## 🛠️ Technologies

- **Backend:** Python, FastAPI, Uvicorn
- **ML/DL:** TensorFlow, Keras, NumPy, Pandas
- **Frontend:** HTML, CSS, JavaScript, Leaflet.js, Chart.js
- **APIs:** Open-Meteo, USGS Earthquake

---

## 📊 Model Architecture

```
Input (Marine + Weather + Seismic Data)
    ↓
┌─────────────┐    ┌─────────────┐
│  CNN Branch │    │ LSTM Branch │
│  (Spatial)  │    │ (Temporal)  │
└──────┬──────┘    └──────┬──────┘
       └────────┬─────────┘
                ↓
         Feature Fusion
                ↓
    ┌───────────┼───────────┐
    ↓           ↓           ↓
Wave Severity  Tsunami   Wave Height
Classification  Risk    Regression
```

---

## 📜 License

MIT License - Feel free to use and modify.

---

**📍 Kanyakumari Ocean Wave & Tsunami Prediction System**
