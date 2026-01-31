import requests

def get_nearby_places_osm(lat, lng, radius=50):
    """Get ALL nearby places from OpenStreetMap - completely free"""
    
    # Overpass API query
    query = f"""
    [out:json];
    (
      node["name"](around:{radius},{lat},{lng});
      way["name"](around:{radius},{lat},{lng});
      relation["name"](around:{radius},{lat},{lng});
    );
    out center;
    """
    
    url = "https://overpass-api.de/api/interpreter"
    response = requests.post(url, data={"data": query}, timeout=30)
    data = response.json()
    
    places = []
    for element in data.get('elements', []):
        tags = element.get('tags', {})
        name = tags.get('name')
        if name:
            place_type = tags.get('building') or tags.get('amenity') or tags.get('tourism') or 'place'
            places.append({
                'name': name,
                'type': place_type
            })
    
    return places

# Test near Lindemann
lat, lng = 41.8276918, -71.4018745
places = get_nearby_places_osm(lat, lng, 50)

print(f"Found {len(places)} places:\n")
for place in places:
    print(f"Name: {place['name']}")
    print(f"Type: {place['type']}")
    print("----")