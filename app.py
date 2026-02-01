from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables from .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=env_path)

from tour_guide_agent import run_tour

app = FastAPI()

# Location request model
class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    heading: float = 0  # Default to facing North

# Tour endpoint that accepts user location
@app.post("/tour")
def get_tour(location: LocationRequest):
    message = run_tour(location.latitude, location.longitude, location.heading)
    return {"message": message}

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