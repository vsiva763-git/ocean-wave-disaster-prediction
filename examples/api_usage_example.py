#!/usr/bin/env python3
"""Example usage of the Ocean Wave Disaster Prediction API.

This script demonstrates how to:
1. Start the API server
2. Make requests to different endpoints
3. Process the responses

Usage:
    # Start the server in the background
    python -m uvicorn api:app --host 0.0.0.0 --port 8000 &
    
    # Run this example script
    python examples/api_usage_example.py
"""

import httpx
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def get_prediction(latitude: float, longitude: float):
    """Get live prediction for a location."""
    print(f"\n{'='*60}")
    print(f"Getting prediction for location: ({latitude}, {longitude})")
    print('='*60)
    
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            f"{BASE_URL}/predict",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "use_ndbc": True,
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Prediction successful!")
            print(f"  Risk Level: {data['predicted_class']}")
            print(f"  Hazard Index: {data['hazard_probability_index']:.4f}")
            print(f"  Confidence: {data['confidence']:.2%}")
            print(f"\n  Probabilities:")
            for class_name, prob in data['probabilities'].items():
                print(f"    {class_name}: {prob:.2%}")
            print(f"\n  Data Sources:")
            print(f"    Open-Meteo: {data['data_sources']['open_meteo']}")
            print(f"    NDBC: {data['data_sources']['ndbc']}")
            return data
        else:
            print(f"✗ Prediction failed: {response.status_code}")
            print(f"  Error: {response.json().get('detail', 'Unknown error')}")
            return None


def get_tsunami_bulletins():
    """Get tsunami warnings and advisories."""
    print(f"\n{'='*60}")
    print("Fetching tsunami bulletins")
    print('='*60)
    
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            f"{BASE_URL}/bulletins",
            params={"active_only": True}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Found {data['count']} active bulletin(s)")
            
            for i, bulletin in enumerate(data['bulletins'][:3], 1):
                print(f"\n  Bulletin {i}:")
                print(f"    Source: {bulletin['source']}")
                print(f"    Severity: {bulletin['severity']}")
                print(f"    Title: {bulletin['title']}")
                print(f"    Published: {bulletin['published']}")
                print(f"    Link: {bulletin['link']}")
            
            return data
        else:
            print(f"✗ Failed to fetch bulletins: {response.status_code}")
            return None


def get_earthquakes(min_magnitude: float = 5.0, tsunami_risk_only: bool = False):
    """Get recent earthquake events."""
    print(f"\n{'='*60}")
    print(f"Fetching earthquakes (M >= {min_magnitude})")
    print('='*60)
    
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            f"{BASE_URL}/earthquakes",
            params={
                "min_magnitude": min_magnitude,
                "hours_back": 168,
                "tsunami_risk_only": tsunami_risk_only,
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Found {data['count']} earthquake(s)")
            
            for i, event in enumerate(data['events'][:5], 1):
                print(f"\n  Event {i}:")
                print(f"    Magnitude: {event['magnitude']:.1f} {event['magnitude_type']}")
                print(f"    Location: {event['place']}")
                print(f"    Depth: {event['depth_km']:.1f} km")
                print(f"    Time: {event['time']}")
                print(f"    Tsunami Flag: {event['tsunami_flag']}")
            
            return data
        else:
            print(f"✗ Failed to fetch earthquakes: {response.status_code}")
            return None


def main():
    """Run all examples."""
    print("="*60)
    print("Ocean Wave Disaster Prediction API - Usage Examples")
    print("="*60)
    print(f"API Base URL: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if API is running
    try:
        response = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        if response.status_code == 200:
            print("✓ API is healthy and running")
        else:
            print("✗ API is not responding correctly")
            return
    except Exception as exc:
        print(f"✗ Cannot connect to API: {exc}")
        print("\nPlease start the API server first:")
        print("  cd src")
        print("  python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000")
        return
    
    # Example locations
    monterey_bay = (36.78, -122.40)
    
    # Example 1: Get prediction
    get_prediction(*monterey_bay)
    
    # Example 2: Get tsunami bulletins
    get_tsunami_bulletins()
    
    # Example 3: Get earthquakes
    get_earthquakes(min_magnitude=5.0, tsunami_risk_only=True)
    
    print(f"\n{'='*60}")
    print("Examples complete!")
    print('='*60)


if __name__ == "__main__":
    main()
