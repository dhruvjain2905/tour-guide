"""
Simple test script to verify the API endpoints work correctly.
Run the FastAPI server first: uvicorn app:app --reload
Then run this script: python test_api.py
"""

import requests
import json
import uuid

BASE_URL = "http://localhost:8000"

def test_create_preferences():
    """Test creating user preferences."""
    print("\n=== Testing POST /preferences/create ===")

    user_id = str(uuid.uuid4())
    data = {
        "user_id": user_id,
        "raw_preferences": "I love history and architecture, especially old buildings with interesting stories"
    }

    response = requests.post(f"{BASE_URL}/preferences/create", json=data)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nUser ID: {result['user_id']}")
        print(f"\nRaw Preferences:\n{result['raw_preferences']}")
        print(f"\nEnhanced Preferences:\n{result['enhanced_preferences']}")
        return user_id
    else:
        print(f"Error: {response.text}")
        return None


def test_get_preferences(user_id):
    """Test getting user preferences."""
    print("\n\n=== Testing GET /preferences/{user_id} ===")

    response = requests.get(f"{BASE_URL}/preferences/{user_id}")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nUser ID: {result['user_id']}")
        print(f"\nRaw Preferences:\n{result['raw_preferences']}")
        print(f"\nEnhanced Preferences (truncated):\n{result['enhanced_preferences'][:200]}...")
        print(f"\nCreated at: {result['created_at']}")
        print(f"Updated at: {result['updated_at']}")
    else:
        print(f"Error: {response.text}")


def test_update_preferences(user_id):
    """Test updating user preferences."""
    print("\n\n=== Testing PUT /preferences/{user_id} ===")

    data = {
        "raw_preferences": "I'm fascinated by local culture, food scenes, and economic history of neighborhoods"
    }

    response = requests.put(f"{BASE_URL}/preferences/{user_id}", json=data)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nUpdated Raw Preferences:\n{result['raw_preferences']}")
        print(f"\nUpdated Enhanced Preferences:\n{result['enhanced_preferences']}")
    else:
        print(f"Error: {response.text}")


def test_tour_without_preferences():
    """Test tour endpoint without creating preferences first."""
    print("\n\n=== Testing POST /tour (without preferences) ===")

    user_id = str(uuid.uuid4())
    data = {
        "user_id": user_id,
        "latitude": 41.8303668,
        "longitude": -71.4015215,
        "heading": 0
    }

    print(f"Using user_id: {user_id} (no preferences set)")
    print("Note: This should use default preferences")

    response = requests.post(f"{BASE_URL}/tour", json=data)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nNarrative (first 200 chars):\n{result['narrative'][:200]}...")
        print(f"\nNumber of places: {len(result['places'])}")
        if result['places']:
            print(f"\nFirst place: {result['places'][0]['name']}")
    else:
        print(f"Error: {response.text}")


def test_tour_with_preferences(user_id):
    """Test tour endpoint with user preferences."""
    print("\n\n=== Testing POST /tour (with preferences) ===")

    data = {
        "user_id": user_id,
        "latitude": 41.8303668,
        "longitude": -71.4015215,
        "heading": 0
    }

    print(f"Using user_id: {user_id} (with preferences)")

    response = requests.post(f"{BASE_URL}/tour", json=data)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nNarrative (first 200 chars):\n{result['narrative'][:200]}...")
        print(f"\nNumber of places: {len(result['places'])}")
        if result['places']:
            print(f"\nFirst place: {result['places'][0]['name']}")
    else:
        print(f"Error: {response.text}")


if __name__ == "__main__":
    print("=" * 60)
    print("API ENDPOINT TESTS")
    print("=" * 60)
    print("\nMake sure the FastAPI server is running:")
    print("  cd tour-guide")
    print("  uvicorn app:app --reload")
    print("\n" + "=" * 60)

    try:
        # Test creating preferences
        user_id = test_create_preferences()

        if user_id:
            # Test getting preferences
            test_get_preferences(user_id)

            # Test updating preferences
            test_update_preferences(user_id)

            # Test tour with preferences (this will take longer due to AI calls)
            # Uncomment to test:
            # test_tour_with_preferences(user_id)

        # Test tour without preferences
        # Uncomment to test:
        # test_tour_without_preferences()

        print("\n\n" + "=" * 60)
        print("TESTS COMPLETED")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the API server.")
        print("Make sure the FastAPI server is running:")
        print("  cd tour-guide")
        print("  uvicorn app:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
