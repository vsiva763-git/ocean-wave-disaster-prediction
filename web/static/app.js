/**
 * Kanyakumari Ocean Wave Prediction System
 * Frontend JavaScript Application
 *
 * Features:
 * - Real-time data fetching and display
 * - Interactive map with Leaflet
 * - Wave forecast chart with Chart.js
 * - Auto-refresh every 5 minutes
 */

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
  API_BASE: window.location.origin,
  KANYAKUMARI: { lat: 8.0883, lon: 77.5385 },
  REFRESH_INTERVAL: 300000, // 5 minutes
  MAP_ZOOM: 8,
  MONITORING_POINTS: {
    kanyakumari: { lat: 8.0883, lon: 77.5385, name: "Kanyakumari" },
    kovalam: { lat: 8.3684, lon: 77.0214, name: "Kovalam" },
    tuticorin: { lat: 8.7642, lon: 78.1348, name: "Tuticorin" },
    rameshwaram: { lat: 9.2876, lon: 79.3129, name: "Rameshwaram" },
    trivandrum: { lat: 8.5074, lon: 76.9558, name: "Thiruvananthapuram" },
  },
};

// ============================================================================
// Global Variables
// ============================================================================

let map = null;
let forecastChart = null;
let refreshTimer = null;
let currentData = null;

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
  console.log("🌊 Kanyakumari Ocean Prediction System - Initializing...");

  initMap();
  initChart();

  // Initial data fetch
  refreshData();

  // Set up auto-refresh
  refreshTimer = setInterval(refreshData, CONFIG.REFRESH_INTERVAL);

  console.log("✅ System initialized");
});

// ============================================================================
// Map Initialization
// ============================================================================

function initMap() {
  // Create map centered on Kanyakumari
  map = L.map("map", {
    center: [CONFIG.KANYAKUMARI.lat, CONFIG.KANYAKUMARI.lon],
    zoom: CONFIG.MAP_ZOOM,
    zoomControl: true,
  });

  // Add dark-themed tile layer
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
    maxZoom: 19,
  }).addTo(map);

  // Add main marker for Kanyakumari
  const mainMarker = L.marker(
    [CONFIG.KANYAKUMARI.lat, CONFIG.KANYAKUMARI.lon],
    {
      icon: createCustomIcon("#ef4444", "🌊"),
    }
  ).addTo(map);

  mainMarker
    .bindPopup(
      `
        <div style="text-align: center; padding: 5px;">
            <strong style="font-size: 14px;">📍 Kanyakumari</strong><br>
            <span style="font-size: 12px; color: #666;">
                Southernmost tip of India<br>
                8.0883°N, 77.5385°E
            </span>
        </div>
    `
    )
    .openPopup();

  // Add monitoring points
  Object.entries(CONFIG.MONITORING_POINTS).forEach(([key, point]) => {
    if (key !== "kanyakumari") {
      const marker = L.circleMarker([point.lat, point.lon], {
        radius: 8,
        fillColor: "#06b6d4",
        color: "#0891b2",
        weight: 2,
        fillOpacity: 0.8,
      }).addTo(map);

      marker.bindPopup(`
                <div style="text-align: center;">
                    <strong>${point.name}</strong><br>
                    <span style="font-size: 11px;">${point.lat.toFixed(
                      4
                    )}°N, ${point.lon.toFixed(4)}°E</span>
                </div>
            `);
    }
  });

  // Add ocean region overlay
  const oceanBounds = [
    [5.0, 74.0],
    [12.0, 82.0],
  ];

  L.rectangle(oceanBounds, {
    color: "#3b82f6",
    weight: 2,
    fillOpacity: 0.05,
    dashArray: "10, 10",
  }).addTo(map);
}

function createCustomIcon(color, emoji) {
  return L.divIcon({
    className: "custom-marker",
    html: `<div style="
            background: ${color};
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            border: 3px solid white;
            box-shadow: 0 3px 10px rgba(0,0,0,0.3);
        ">${emoji}</div>`,
    iconSize: [40, 40],
    iconAnchor: [20, 20],
  });
}

// ============================================================================
// Chart Initialization
// ============================================================================

function initChart() {
  const ctx = document.getElementById("forecast-chart");
  if (!ctx) return;

  forecastChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Wave Height (m)",
          data: [],
          borderColor: "#06b6d4",
          backgroundColor: "rgba(6, 182, 212, 0.1)",
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 5,
        },
        {
          label: "Swell Height (m)",
          data: [],
          borderColor: "#8b5cf6",
          backgroundColor: "rgba(139, 92, 246, 0.1)",
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: {
          display: true,
          position: "top",
          labels: {
            color: "#a0aec0",
            usePointStyle: true,
            padding: 15,
          },
        },
        tooltip: {
          backgroundColor: "#132743",
          titleColor: "#fff",
          bodyColor: "#a0aec0",
          borderColor: "#1e3a5f",
          borderWidth: 1,
          padding: 12,
          displayColors: true,
        },
      },
      scales: {
        x: {
          display: true,
          grid: {
            color: "rgba(30, 58, 95, 0.5)",
            drawBorder: false,
          },
          ticks: {
            color: "#718096",
            maxTicksLimit: 12,
          },
        },
        y: {
          display: true,
          beginAtZero: true,
          grid: {
            color: "rgba(30, 58, 95, 0.5)",
            drawBorder: false,
          },
          ticks: {
            color: "#718096",
            callback: (value) => value + "m",
          },
        },
      },
    },
  });
}

// ============================================================================
// Data Fetching
// ============================================================================

async function refreshData() {
  console.log("🔄 Refreshing data...");

  try {
    // Fetch current conditions and prediction
    const response = await fetch(
      `${CONFIG.API_BASE}/api/predict?include_forecast=true&forecast_hours=48`
    );

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    currentData = data;

    // Update all UI components
    updateCurrentConditions(data);
    updatePrediction(data.prediction);
    updateForecastChart(data.forecast);
    updateRiskAssessment(data.overall_risk);
    updateTimestamp();

    // Fetch earthquake data separately
    fetchEarthquakes();

    console.log("✅ Data refresh complete");
  } catch (error) {
    console.error("❌ Error fetching data:", error);
    showAlert(`Data fetch failed: ${error.message}`);
  }
}

async function fetchEarthquakes() {
  try {
    const response = await fetch(
      `${CONFIG.API_BASE}/api/earthquakes?min_magnitude=5&hours_back=168`
    );

    if (response.ok) {
      const data = await response.json();
      updateEarthquakesList(data);
    }
  } catch (error) {
    console.error("Error fetching earthquakes:", error);
  }
}

// ============================================================================
// UI Update Functions
// ============================================================================

function updateCurrentConditions(data) {
  const current = data.current_conditions || {};
  const marine = current.marine || {};
  const weather = current.weather || {};

  // Wave conditions
  updateElement("wave-height-value", formatValue(marine.wave_height_m, 1));
  updateElement("wave-condition", marine.wave_condition_display || "Unknown");
  updateElement("wave-period", formatValue(marine.wave_period_s, 1) + " s");
  updateElement("wave-direction", marine.wave_direction_name || "--");
  updateElement("swell-height", formatValue(marine.swell_height_m, 1) + " m");

  // Weather conditions
  updateElement("temperature-value", formatValue(weather.temperature_c, 0));
  updateElement("weather-desc", weather.conditions || "Unknown");
  updateElement("wind-speed-value", formatValue(weather.wind_speed_kmh, 0));
  updateElement("wind-direction", weather.wind_direction || "--");
  updateElement("humidity", formatValue(weather.humidity_percent, 0) + "%");
  updateElement("pressure", formatValue(weather.pressure_hpa, 0) + " hPa");

  // Max wave forecast
  const maxWave = data.max_wave_forecast || {};
  updateElement("max-wave-height", formatValue(maxWave.height_m, 1) + " m");

  if (maxWave.expected_time) {
    const maxTime = new Date(maxWave.expected_time);
    updateElement(
      "max-wave-time",
      maxTime.toLocaleString("en-IN", {
        hour: "2-digit",
        minute: "2-digit",
        day: "numeric",
        month: "short",
      })
    );
  }
}

function updatePrediction(prediction) {
  if (!prediction) return;

  // Wave severity
  const severity = String(prediction.wave_severity || "NORMAL");
  const severityElement = document.getElementById("wave-severity");
  if (severityElement) {
    severityElement.textContent = severity;
    severityElement.className = "severity-badge " + severity.toLowerCase();
  }

  // Tsunami risk
  const tsunamiRisk = String(prediction.tsunami_risk || "NONE");
  const riskElement = document.getElementById("tsunami-risk");
  if (riskElement) {
    riskElement.textContent = tsunamiRisk;
    riskElement.className = "risk-badge " + tsunamiRisk.toLowerCase();
  }

  // Hazard index
  const hazardIndex = Number(prediction.hazard_index || 0);
  updateElement("hazard-index-value", hazardIndex.toFixed(3));

  // Confidence
  const confidence = Number(prediction.confidence || 0) * 100;
  updateElement("confidence-value", confidence.toFixed(0) + "%");
  const confidenceFill = document.getElementById("confidence-fill");
  if (confidenceFill) {
    confidenceFill.style.width = confidence + "%";
  }

  // Wave probabilities
  const waveProbs = prediction.wave_probabilities || {};
  updateProbabilityBar("prob-normal", waveProbs.NORMAL || 0);
  updateProbabilityBar("prob-moderate", waveProbs.MODERATE || 0);
  updateProbabilityBar("prob-high", waveProbs.HIGH || 0);
  updateProbabilityBar("prob-extreme", waveProbs.EXTREME || 0);

  // Tsunami probabilities
  const tsunamiProbs = prediction.tsunami_probabilities || {};
  updateProbabilityBar("prob-tsunami-none", tsunamiProbs.NONE || 0);
  updateProbabilityBar("prob-tsunami-low", tsunamiProbs.LOW || 0);
  updateProbabilityBar("prob-tsunami-high", tsunamiProbs.HIGH || 0);

  // Show alert for high risk
  if (severity === "EXTREME" || tsunamiRisk === "HIGH") {
    showAlert(
      `⚠️ ${
        severity === "EXTREME"
          ? "EXTREME wave conditions detected!"
          : "HIGH tsunami risk detected!"
      }`
    );
  } else {
    hideAlert();
  }
}

function updateProbabilityBar(elementId, value) {
  const bar = document.getElementById(elementId);
  const valueEl = document.getElementById(elementId + "-val");

  if (bar) {
    bar.style.width = value * 100 + "%";
  }
  if (valueEl) {
    valueEl.textContent = (value * 100).toFixed(1) + "%";
  }
}

function updateForecastChart(forecast) {
  if (!forecastChart || !forecast || !forecast.length) return;

  const labels = [];
  const waveHeights = [];
  const swellHeights = [];

  forecast.forEach((point, index) => {
    if (index % 2 === 0) {
      // Show every 2 hours
      const time = new Date(point.time);
      labels.push(
        time.toLocaleString("en-IN", {
          hour: "2-digit",
          day: "numeric",
          month: "short",
        })
      );
      waveHeights.push(point.wave_height || 0);
      swellHeights.push(point.swell_wave_height || 0);
    }
  });

  forecastChart.data.labels = labels;
  forecastChart.data.datasets[0].data = waveHeights;
  forecastChart.data.datasets[1].data = swellHeights;
  forecastChart.update("none");
}

function updateEarthquakesList(data) {
  const listElement = document.getElementById("earthquakes-list");
  const countElement = document.getElementById("earthquake-count");

  if (!listElement) return;

  const events = data.events || [];

  // Update count badge
  if (countElement) {
    countElement.textContent = `${events.length} events`;
  }

  if (events.length === 0) {
    listElement.innerHTML = `
            <div class="loading-placeholder">
                ✅ No significant seismic activity in the last 7 days
            </div>
        `;
    return;
  }

  // Show top 5 events
  const html = events
    .slice(0, 5)
    .map((event) => {
      const magClass =
        event.magnitude >= 7
          ? "high"
          : event.magnitude >= 6
          ? "moderate"
          : "low";
      const riskClass = event.tsunami_risk_level || "none";

      return `
            <div class="earthquake-item">
                <div class="eq-magnitude ${magClass}">${
        event.magnitude?.toFixed(1) || "--"
      }</div>
                <div class="eq-details">
                    <div class="eq-location">${
                      event.place || "Unknown location"
                    }</div>
                    <div class="eq-info">
                        <span>📏 ${
                          event.distance_km?.toFixed(0) || "--"
                        } km away</span>
                        <span>📅 ${formatDate(event.time)}</span>
                    </div>
                </div>
                <div class="eq-risk ${riskClass}">${riskClass.toUpperCase()}</div>
            </div>
        `;
    })
    .join("");

  listElement.innerHTML = html;
}

function updateRiskAssessment(risk) {
  if (!risk) return;

  const level = risk.risk_level || "none";
  const score = risk.risk_score || 0;
  const recommendation = risk.recommendation || "Normal conditions.";

  // Update risk indicator
  const indicator = document.getElementById("overall-risk-indicator");
  const levelElement = document.getElementById("overall-risk-level");

  if (indicator) {
    indicator.className = "risk-indicator " + level;
  }
  if (levelElement) {
    levelElement.textContent = level.toUpperCase();
  }

  // Update score
  updateElement("risk-score", score.toFixed(2));

  // Update recommendation
  updateElement("recommendation", recommendation);
}

function updateTimestamp() {
  const timeElement = document.getElementById("update-time");
  if (timeElement) {
    const now = new Date();
    timeElement.textContent = `Last Updated: ${now.toLocaleString("en-IN", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      day: "numeric",
      month: "short",
    })}`;
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

function updateElement(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = value;
  }
}

function formatValue(value, decimals = 1) {
  if (value === null || value === undefined) return "--";
  return Number(value).toFixed(decimals);
}

function formatDate(dateString) {
  if (!dateString) return "--";
  const date = new Date(dateString);
  return date.toLocaleString("en-IN", {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function showAlert(message) {
  const banner = document.getElementById("alert-banner");
  const messageEl = document.getElementById("alert-message");

  if (banner && messageEl) {
    messageEl.textContent = message;
    banner.style.display = "block";
  }
}

function hideAlert() {
  const banner = document.getElementById("alert-banner");
  if (banner) {
    banner.style.display = "none";
  }
}

// ============================================================================
// Export Functions for HTML
// ============================================================================

window.refreshData = refreshData;
