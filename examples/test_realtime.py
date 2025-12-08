#!/usr/bin/env python3
"""Manual test script for real-time data fetchers."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from realtime.open_meteo import fetch_open_meteo_marine, prepare_wave_features
from realtime.ndbc import find_nearest_station, fetch_ndbc_latest, prepare_ndbc_features
from realtime.usgs_earthquake import fetch_usgs_earthquakes, filter_tsunami_risk_events
from realtime.tsunami_bulletins import fetch_tsunami_bulletins, get_active_warnings

def test_open_meteo():
    """Test Open-Meteo Marine API."""
    print("\n" + "="*60)
    print("Testing Open-Meteo Marine API")
    print("="*60)
    
    # Monterey Bay, CA
    lat, lon = 36.78, -122.40
    print(f"Location: {lat}, {lon} (Monterey Bay, CA)")
    
    data = fetch_open_meteo_marine(lat, lon, hours=24)
    if not data.empty:
        print(f"✓ Fetched {len(data)} hourly forecasts")
        print(f"  Columns: {list(data.columns)}")
        print(f"  First row:\n{data.head(1).to_string()}")
        
        features = prepare_wave_features(data)
        print(f"✓ Prepared {len(features.columns)} model features: {list(features.columns)}")
    else:
        print("✗ No data returned")

def test_ndbc():
    """Test NDBC buoy data."""
    print("\n" + "="*60)
    print("Testing NDBC Buoy Data")
    print("="*60)
    
    # Try Monterey Bay area
    lat, lon = 36.78, -122.40
    print(f"Finding nearest station to {lat}, {lon}")
    
    station = find_nearest_station(lat, lon, max_distance_km=500)
    if station:
        print(f"✓ Found station: {station}")
        
        data = fetch_ndbc_latest(station, data_type="txt")
        if not data.empty:
            print(f"✓ Fetched {len(data)} observations")
            print(f"  Columns: {list(data.columns)[:10]}...")  # Show first 10
            print(f"  Latest observation:\n{data.tail(1).to_string()}")
            
            features = prepare_ndbc_features(data, data_type="txt")
            print(f"✓ Prepared model features: {list(features.columns)}")
        else:
            print("✗ No buoy data available")
    else:
        print("✗ No station found within range")

def test_usgs_earthquakes():
    """Test USGS earthquake catalog."""
    print("\n" + "="*60)
    print("Testing USGS Earthquake Catalog")
    print("="*60)
    
    print("Fetching earthquakes >= 5.0 magnitude from last 7 days...")
    earthquakes = fetch_usgs_earthquakes(min_magnitude=5.0, hours_back=168)
    
    if not earthquakes.empty:
        print(f"✓ Found {len(earthquakes)} earthquakes")
        print(f"  Columns: {list(earthquakes.columns)}")
        
        # Show recent large events
        print("\nRecent events:")
        for idx, row in earthquakes.tail(5).iterrows():
            print(f"  M{row['magnitude']:.1f} - {row['place']} - {idx.strftime('%Y-%m-%d %H:%M')}")
        
        # Filter for tsunami risk
        tsunami_risk = filter_tsunami_risk_events(earthquakes, min_magnitude=6.5)
        print(f"\n✓ {len(tsunami_risk)} events with tsunami potential (M>=6.5, depth<=70km)")
    else:
        print("✗ No earthquakes found (or API unavailable)")

def test_tsunami_bulletins():
    """Test tsunami bulletin fetchers."""
    print("\n" + "="*60)
    print("Testing Tsunami Bulletins (PTWC/NTWC)")
    print("="*60)
    
    try:
        bulletins = fetch_tsunami_bulletins(sources=["ptwc", "ntwc"])
        print(f"✓ Fetched {len(bulletins)} bulletins")
        
        if bulletins:
            print("\nMost recent bulletin:")
            latest = bulletins[0]
            print(f"  Source: {latest['source']}")
            print(f"  Severity: {latest['severity']}")
            print(f"  Title: {latest['title']}")
            print(f"  Published: {latest['published']}")
            
            active = get_active_warnings(bulletins)
            if active:
                print(f"\n⚠️  {len(active)} active warning(s)/advisory(s)")
            else:
                print(f"\n✓ No active warnings")
        else:
            print("  No bulletins available")
    except Exception as exc:
        print(f"✗ Failed: {exc}")

if __name__ == "__main__":
    print("="*60)
    print("Real-time Data Fetchers Test Suite")
    print("="*60)
    
    test_open_meteo()
    test_ndbc()
    test_usgs_earthquakes()
    test_tsunami_bulletins()
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)
