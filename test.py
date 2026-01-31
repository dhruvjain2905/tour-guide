import requests
import json
import os

api_key = os.getenv("GOOGLE_API_KEY")

# New API endpoint
url = "https://places.googleapis.com/v1/places:searchNearby"

# Headers (required for new API)
headers = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": api_key,
    "X-Goog-FieldMask": (
        "places.displayName,"
        "places.formattedAddress,"
        "places.types,"
        "places.location"
    )
}

# Request body
body = {
    "locationRestriction": {
        "circle": {
            "center": {
                "latitude": 41.8261395,    
                "longitude": -71.4045595
            },
            "radius": 80.0
        }
    },
}

# Make the request
response = requests.post(url, headers=headers, json=body)
data = response.json()

print(data)
print("\n" + "="*50 + "\n")

# Extract and print information
if "places" in data:
    for place in data["places"]:
        print(f"Name: {place['displayName']['text']}")
        print(f"Address: {place.get('formattedAddress', 'N/A')}")
        print(f"Types: {', '.join(place.get('types', []))}")

        if "location" in place:
            lat = place["location"]["latitude"]
            lng = place["location"]["longitude"]
            print(f"Coordinates: ({lat}, {lng})")
        else:
            print("Coordinates: N/A")

        print("----")
else:
    print("No places found or error occurred")
    print(data)