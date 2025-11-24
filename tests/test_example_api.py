import os
os.environ["API_KEYS"] = "test_api_key"
os.environ["HOST"] = "0.0.0.0"
os.environ["PORT"] = "31256"

import requests
import sys
import server.app
from fastapi.testclient import TestClient

def test_get_example(): 
    client = TestClient(server.app.app)

    filename = "sample11_Normal.csv"
    
    print(f"Testing GET /examples/{filename}...")
    
    try:
        response = client.get(f"/examples/{filename}")
        
        if response.status_code == 200:
            data = response.json()
            print("Success! Response received.")
            print(f"Filename: {data.get('filename')}")
            signal = data.get('signal')
            if signal and isinstance(signal, list) and len(signal) > 0:
                print(f"Signal received. Length: {len(signal)}")
                print(f"First 5 values: {signal[:5]}")
            else:
                print("Error: Signal data missing or invalid.")
                sys.exit(1)
        else:
            print(f"Failed. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Exception occurred: {e}")
        sys.exit(1)

    # Test invalid file
    invalid_filename = "non_existent_file.csv"
    print(f"\nTesting GET /examples/{invalid_filename} (expecting 404)...")
    try:
        response = client.get(f"/examples/{invalid_filename}")
        if response.status_code == 404:
            print("Success! Received 404 as expected.")
        else:
            print(f"Failed. Expected 404, got {response.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"Exception occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_get_example()
