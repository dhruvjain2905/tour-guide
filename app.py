from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables from .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=env_path)

from tour_guide_agent_withcontext import run_tour
from preferences import enhance_preferences, get_default_preferences
from database import (
    create_user_preferences,
    get_user_preferences,
    update_user_preferences
)

app = FastAPI()

# Preference request models
class PreferenceCreateRequest(BaseModel):
    user_id: str
    raw_preferences: str

class PreferenceUpdateRequest(BaseModel):
    raw_preferences: str

# Location request model
class LocationRequest(BaseModel):
    user_id: str
    latitude: float
    longitude: float
    heading: float = 0  # Default to facing North
    previous_summary: str | None = None  # Summary from previous tour segment for continuation


# Preference endpoints
@app.post("/preferences/create")
def create_preferences(request: PreferenceCreateRequest):
    """
    Create user preferences by enhancing their brief input with AI.
    """
    # Check if user already has preferences
    existing = get_user_preferences(request.user_id)
    if existing:
        raise HTTPException(status_code=400, detail="User preferences already exist. Use PUT to update.")

    # Enhance the preferences using AI
    try:
        enhanced = enhance_preferences(request.raw_preferences)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enhance preferences: {str(e)}")

    # Save to database
    success = create_user_preferences(request.user_id, request.raw_preferences, enhanced)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save preferences")

    return {
        "user_id": request.user_id,
        "raw_preferences": request.raw_preferences,
        "enhanced_preferences": enhanced
    }


@app.get("/preferences/{user_id}")
def get_preferences(user_id: str):
    """
    Get user preferences by user_id.
    """
    prefs = get_user_preferences(user_id)
    if not prefs:
        raise HTTPException(status_code=404, detail="User preferences not found")

    return {
        "user_id": prefs["user_id"],
        "raw_preferences": prefs["raw_preferences"],
        "enhanced_preferences": prefs["enhanced_preferences"],
        "created_at": prefs["created_at"],
        "updated_at": prefs["updated_at"]
    }


@app.put("/preferences/{user_id}")
def update_preferences(user_id: str, request: PreferenceUpdateRequest):
    """
    Update user preferences with new input (will be re-enhanced by AI).
    """
    # Check if user exists
    existing = get_user_preferences(user_id)
    if not existing:
        raise HTTPException(status_code=404, detail="User preferences not found")

    # Enhance the new preferences using AI
    try:
        enhanced = enhance_preferences(request.raw_preferences)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enhance preferences: {str(e)}")

    # Update in database
    success = update_user_preferences(user_id, request.raw_preferences, enhanced)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update preferences")

    return {
        "user_id": user_id,
        "raw_preferences": request.raw_preferences,
        "enhanced_preferences": enhanced
    }


# Tour endpoint that accepts user location
@app.post("/tour")
def get_tour(location: LocationRequest):
    # Fetch user preferences from database
    prefs = get_user_preferences(location.user_id)

    # Use enhanced preferences if available, otherwise use defaults
    if prefs:
        user_preferences = prefs["enhanced_preferences"]
    else:
        user_preferences = get_default_preferences()
        print(f"Warning: No preferences found for user {location.user_id}, using defaults")

    result = run_tour(
        latitude=location.latitude,
        longitude=location.longitude,
        heading=location.heading,
        previous_summary=location.previous_summary,
        stream=False,
        user_preferences=user_preferences  # Pass preferences to tour agent
    )

    # Extract tour output and convert to JSON-serializable format
    tour_output = result["tour"]
    places = [
        {
            "name": place.name,
            "latitude": place.latitude,
            "longitude": place.longitude,
            "relative_direction": place.relative_direction,
            "distance_meters": place.distance_meters
        }
        for place in tour_output.places
    ]

    return {
        "narrative": tour_output.narrative,
        "places": places,
        "summary": result["summary"],
        "latitude": location.latitude,
        "longitude": location.longitude,
        "heading": location.heading
    }

# Example POST route
class AddRequest(BaseModel):
    a: int
    b: int

@app.post("/add")
def add_numbers(request: AddRequest):
    result = request.a + request.b
    return {"result": result}


# if __name__ == "__main__":
#     main()