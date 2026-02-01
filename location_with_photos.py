import requests
import json
import os

user_preferences = """I'm deeply fascinated by the history of places I visit.
I love learning about both iconic landmarks and hidden gems with compelling backstories - sometimes the lesser-known buildings have the most interesting tales.
I'm particularly drawn to understanding how individual sites connect to broader historical narratives: how did this place fit into larger events?
 What was happening here 50, 100, or 200 years ago? How has the neighborhood transformed over different eras?

Beyond individual buildings, I'm curious about the economic evolution of areas.
What industries drove this neighborhood's development? What brought people here originally? I want to know about current economic activity too - what businesses and industries thrive here now?
Are there any unique local enterprises or economic quirks that make this area special?

I also appreciate learning about the cultural and social fabric of a place.
What makes this area distinctive today? Are there interesting creative communities, tech hubs, or artisan scenes? What draws people to live and work here now versus historically?
I love discovering the "cool factor" - whether that's innovative architecture, vibrant street art, unique local shops, or community gathering spaces that give a place its character.
Essentially, I want to understand both the "then" and the "now" - how history shaped what I'm seeing, and what makes this place dynamic and interesting in the present day."""


def get_photo_url(photo_name, api_key, max_height=400, max_width=400):
    """
    Construct the photo URL from a photo resource name.

    Args:
        photo_name: The photo resource name from the places API (e.g., "places/ChIJ.../photos/...")
        api_key: Your Google API key
        max_height: Maximum height in pixels (1-4800)
        max_width: Maximum width in pixels (1-4800)

    Returns:
        URL string that will redirect to the actual image
    """
    return f"https://places.googleapis.com/v1/{photo_name}/media?key={api_key}&maxHeightPx={max_height}&maxWidthPx={max_width}"


def get_places(latitude, longitude, radius=80.0):
    """Get nearby places using Google Places API, including photos"""

    api_key = "AIzaSyAYCqxtkoGqRxe_ec1U_2sUzO01-BBo8Jw"

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
            "places.location,"
            "places.photos"
        )
    }

    # Request body
    body = {
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": latitude,
                    "longitude": longitude
                },
                "radius": radius
            }
        },
        "maxResultCount": 20
    }

    # Make the request
    response = requests.post(url, headers=headers, json=body)
    data = response.json()

    # Debug: print raw response
    #print("API Response Status:", response.status_code)
    #print("API Response:", json.dumps(data, indent=2))

    # Extract and format the places data
    places = []
    if "places" in data:
        for place in data["places"]:
            # Get the first photo URL if photos exist
            photo_url = None
            photos = place.get('photos', [])
            if photos:
                # Get the first photo's resource name
                photo_name = photos[0].get('name')
                if photo_name:
                    photo_url = get_photo_url(photo_name, api_key)

            place_info = {
                "name": place['displayName']['text'],
                "address": place.get('formattedAddress', 'N/A'),
                "types": place.get('types', []),
                "latitude": place.get('location', {}).get('latitude', None),
                "longitude": place.get('location', {}).get('longitude', None),
                "photo_url": photo_url
            }
            places.append(place_info)

    return places


places = get_places(41.8261395, -71.4045595, 80.0)

for i in places:
    print(i)
    print("-"*20)