// Configuration
const API_BASE_URL = window.location.origin;
const DEFAULT_CENTER = [20.5937, 78.9629]; // India center
const DEFAULT_ZOOM = 4;

// Location presets
const LOCATION_PRESETS = {
    'bay-of-bengal': { lat: 15.0, lon: 88.0, zoom: 5, name: 'Bay of Bengal' },
    'arabian-sea': { lat: 15.0, lon: 65.0, zoom: 5, name: 'Arabian Sea' },
    'pacific-ocean': { lat: 0.0, lon: -140.0, zoom: 3, name: 'Pacific Ocean' },
    'atlantic-ocean': { lat: 30.0, lon: -40.0, zoom: 3, name: 'Atlantic Ocean' },
    'indian-ocean': { lat: -20.0, lon: 75.0, zoom: 4, name: 'Indian Ocean' }
};

// Global variables
let map;
let currentMarker;
let selectedLocation = null;

// Initialize map on page load
document.addEventListener('DOMContentLoaded', function() {
    initMap();
    console.log('Ocean Wave Disaster Prediction System initialized');
});

// Initialize Leaflet map
function initMap() {
    map = L.map('map').setView(DEFAULT_CENTER, DEFAULT_ZOOM);
    
    // Add OpenStreetMap tile layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 18
    }).addTo(map);
    
    // Add click event listener
    map.on('click', function(e) {
        const lat = e.latlng.lat.toFixed(2);
        const lon = e.latlng.lng.toFixed(2);
        selectLocation(lat, lon);
    });
}

// Select location preset
function selectPreset(presetId) {
    const preset = LOCATION_PRESETS[presetId];
    if (!preset) return;
    
    // Update map view
    map.setView([preset.lat, preset.lon], preset.zoom);
    
    // Select the location
    selectLocation(preset.lat, preset.lon, preset.name);
}

// Select a location on the map
function selectLocation(lat, lon, name = null) {
    lat = parseFloat(lat);
    lon = parseFloat(lon);
    
    // Update input fields
    document.getElementById('latitude').value = lat;
    document.getElementById('longitude').value = lon;
    
    // Update marker
    if (currentMarker) {
        map.removeLayer(currentMarker);
    }
    currentMarker = L.marker([lat, lon]).addTo(map);
    currentMarker.bindPopup(`<b>${name || 'Selected Location'}</b><br>Lat: ${lat}, Lon: ${lon}`).openPopup();
    
    // Store selected location
    selectedLocation = { lat, lon, name };
    
    // Automatically get prediction
    getPrediction();
}

// Get prediction for selected location
async function getPrediction() {
    const lat = parseFloat(document.getElementById('latitude').value);
    const lon = parseFloat(document.getElementById('longitude').value);
    
    if (isNaN(lat) || isNaN(lon)) {
        showError('Please enter valid coordinates or select a location on the map');
        return;
    }
    
    // Validate coordinate ranges
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
        showError('Coordinates out of range. Latitude: -90 to 90, Longitude: -180 to 180');
        return;
    }
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('prediction-results').style.display = 'none';
    document.getElementById('error-message').style.display = 'none';
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict?latitude=${lat}&longitude=${lon}&use_ndbc=true`);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to fetch prediction');
        }
        
        const data = await response.json();
        displayPrediction(data);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showError(`Error: ${error.message}`);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

// Display prediction results
function displayPrediction(data) {
    // Update location info
    const locationName = selectedLocation?.name || 'Selected Location';
    document.getElementById('location-name').textContent = locationName;
    document.getElementById('coordinates').textContent = 
        `Latitude: ${data.location.latitude}°, Longitude: ${data.location.longitude}°`;
    
    // Update hazard level badge
    const hazardBadge = document.getElementById('hazard-badge');
    hazardBadge.textContent = data.predicted_class;
    hazardBadge.className = 'badge ' + data.predicted_class.toLowerCase();
    
    // Update HPI
    const hpi = data.hazard_probability_index;
    document.getElementById('hpi-fill').style.width = (hpi * 100) + '%';
    document.getElementById('hpi-value').textContent = hpi.toFixed(3);
    
    // Update probabilities
    const probs = data.probabilities;
    updateProbabilityBar('normal', probs.NORMAL);
    updateProbabilityBar('moderate', probs.MODERATE);
    updateProbabilityBar('giant', probs.GIANT);
    
    // Update data sources
    const sourcesInfo = document.getElementById('sources-info');
    sourcesInfo.innerHTML = `
        <p>✓ Open-Meteo Marine API: ${data.data_sources.open_meteo ? 'Active' : 'Unavailable'}</p>
        <p>✓ NDBC Buoy Data: ${data.data_sources.ndbc ? 'Active' : 'Unavailable'}</p>
    `;
    
    // Update timestamp
    const timestamp = new Date(data.timestamp).toLocaleString();
    document.getElementById('timestamp').textContent = `Last updated: ${timestamp}`;
    
    // Show results
    document.getElementById('prediction-results').style.display = 'block';
}

// Update probability bar
function updateProbabilityBar(type, value) {
    const percentage = (value * 100).toFixed(1);
    document.getElementById(`prob-${type}`).style.width = percentage + '%';
    document.getElementById(`prob-${type}-val`).textContent = percentage + '%';
}

// Show error message
function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    document.getElementById('loading').style.display = 'none';
    document.getElementById('prediction-results').style.display = 'none';
}

// Tab switching
function showTab(tabName, event) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => content.classList.remove('active'));
    
    // Remove active class from all tab buttons
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => btn.classList.remove('active'));
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Highlight active button (if event provided)
    if (event && event.target) {
        event.target.classList.add('active');
    }
}

// Get earthquake data
async function getEarthquakes() {
    const minMagnitude = parseFloat(document.getElementById('min-magnitude').value) || 5.0;
    const tsunamiOnly = document.getElementById('tsunami-only').checked;
    
    const listDiv = document.getElementById('earthquakes-list');
    listDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Loading earthquake data...</p></div>';
    
    try {
        let url = `${API_BASE_URL}/earthquakes?min_magnitude=${minMagnitude}&hours_back=24`;
        if (tsunamiOnly) {
            url += '&tsunami_risk_only=true';
        }
        
        // Add target location if selected
        if (selectedLocation) {
            url += `&target_lat=${selectedLocation.lat}&target_lon=${selectedLocation.lon}`;
        }
        
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error('Failed to fetch earthquake data');
        }
        
        const data = await response.json();
        displayEarthquakes(data);
        
    } catch (error) {
        console.error('Earthquake fetch error:', error);
        listDiv.innerHTML = `<p class="error-message">Error: ${error.message}</p>`;
    }
}

// Display earthquake data
function displayEarthquakes(data) {
    const listDiv = document.getElementById('earthquakes-list');
    
    if (data.count === 0) {
        listDiv.innerHTML = '<p>No earthquakes found matching the criteria.</p>';
        return;
    }
    
    let html = `<p><strong>Found ${data.count} earthquake(s) in the last 24 hours</strong></p>`;
    
    data.events.forEach(event => {
        const time = new Date(event.time).toLocaleString();
        const tsunamiFlag = event.tsunami_flag === 1 ? 
            '<span class="tsunami-flag">⚠️ TSUNAMI RISK</span>' : '';
        const eta = event.tsunami_eta ? 
            `<p><strong>Estimated Tsunami Arrival:</strong> ${event.tsunami_eta.eta_minutes.toFixed(0)} minutes (${event.tsunami_eta.distance_km.toFixed(0)} km away)</p>` : '';
        
        html += `
            <div class="event-item">
                <div class="event-header">
                    <span class="event-magnitude">M${event.magnitude}</span>
                    ${tsunamiFlag}
                </div>
                <div class="event-location">${event.place}</div>
                <div class="event-details">
                    <p>Location: ${event.latitude.toFixed(2)}°, ${event.longitude.toFixed(2)}°</p>
                    <p>Depth: ${event.depth_km.toFixed(1)} km</p>
                    ${eta}
                </div>
                <div class="event-time">${time}</div>
                <a href="${event.url}" target="_blank" style="font-size: 0.85em; color: #667eea;">View Details →</a>
            </div>
        `;
    });
    
    listDiv.innerHTML = html;
}

// Get tsunami bulletins
async function getBulletins() {
    const activeOnly = document.getElementById('active-only').checked;
    
    const listDiv = document.getElementById('bulletins-list');
    listDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Loading tsunami bulletins...</p></div>';
    
    try {
        let url = `${API_BASE_URL}/bulletins?sources=ptwc,ntwc`;
        if (activeOnly) {
            url += '&active_only=true';
        }
        
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error('Failed to fetch tsunami bulletins');
        }
        
        const data = await response.json();
        displayBulletins(data);
        
    } catch (error) {
        console.error('Bulletins fetch error:', error);
        listDiv.innerHTML = `<p class="error-message">Error: ${error.message}</p>`;
    }
}

// Display tsunami bulletins
function displayBulletins(data) {
    const listDiv = document.getElementById('bulletins-list');
    
    if (data.count === 0) {
        listDiv.innerHTML = '<p>No tsunami bulletins found. ✓ All clear!</p>';
        return;
    }
    
    let html = `<p><strong>Found ${data.count} bulletin(s)</strong></p>`;
    
    data.bulletins.forEach(bulletin => {
        const time = new Date(bulletin.published).toLocaleString();
        const severityClass = bulletin.severity.toLowerCase();
        
        html += `
            <div class="bulletin-item ${severityClass}">
                <span class="bulletin-severity ${severityClass}">${bulletin.severity}</span>
                <div class="bulletin-title">${bulletin.title}</div>
                <div class="bulletin-summary">${bulletin.summary}</div>
                <div class="bulletin-link">
                    <a href="${bulletin.link}" target="_blank">Read full bulletin →</a>
                </div>
                <div class="event-time">Published: ${time} | Source: ${bulletin.source.toUpperCase()}</div>
            </div>
        `;
    });
    
    listDiv.innerHTML = html;
}

// Auto-refresh data every 5 minutes (optional)
function startAutoRefresh() {
    setInterval(() => {
        if (selectedLocation) {
            console.log('Auto-refreshing prediction data...');
            getPrediction();
        }
    }, 5 * 60 * 1000); // 5 minutes
}

// Uncomment to enable auto-refresh
// startAutoRefresh();
