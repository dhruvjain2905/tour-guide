import os
import math
import json
import time
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool, StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from location_claude import get_places, user_preferences


load_dotenv()  # Load environment variables from .env file


# --- Helper Functions ---

def calculate_relative_direction(user_lat: float, user_lon: float, user_heading: float,
                                  place_lat: float, place_lon: float) -> tuple[str, float]:
    """
    Calculate the relative direction and distance from user to a place.

    Args:
        user_lat, user_lon: User's coordinates
        user_heading: User's facing direction in degrees (0=North, 90=East, etc.)
        place_lat, place_lon: Place's coordinates

    Returns:
        (direction_description, distance_in_meters)
    """
    # Calculate bearing from user to place
    lat1 = math.radians(user_lat)
    lat2 = math.radians(place_lat)
    lon_diff = math.radians(place_lon - user_lon)

    x = math.sin(lon_diff) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon_diff)

    bearing = math.degrees(math.atan2(x, y))
    bearing = (bearing + 360) % 360  # Normalize to 0-360

    # Calculate relative angle from user's heading
    relative_angle = (bearing - user_heading + 360) % 360

    # Convert to direction description
    if relative_angle <= 22.5 or relative_angle > 337.5:
        direction = "directly ahead"
    elif 22.5 < relative_angle <= 67.5:
        direction = "ahead and to your right"
    elif 67.5 < relative_angle <= 112.5:
        direction = "to your right"
    elif 112.5 < relative_angle <= 157.5:
        direction = "behind you and to your right"
    elif 157.5 < relative_angle <= 202.5:
        direction = "behind you"
    elif 202.5 < relative_angle <= 247.5:
        direction = "behind you and to your left"
    elif 247.5 < relative_angle <= 292.5:
        direction = "to your left"
    else:  # 292.5 < relative_angle <= 337.5
        direction = "ahead and to your left"

    # Calculate distance using Haversine formula
    R = 6371000  # Earth's radius in meters
    dlat = math.radians(place_lat - user_lat)
    dlon = math.radians(place_lon - user_lon)
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c

    return direction, round(distance, 1)


def get_heading_description(heading: float) -> str:
    """Convert heading degrees to cardinal direction description."""
    directions = [
        (0, "North"), (45, "Northeast"), (90, "East"), (135, "Southeast"),
        (180, "South"), (225, "Southwest"), (270, "West"), (315, "Northwest"), (360, "North")
    ]
    for deg, name in directions:
        if abs(heading - deg) <= 22.5:
            return name
    return "North"


# --- Tool Factories ---
# These create tools with user location bound, so the agent doesn't need to pass coordinates repeatedly

def create_search_nearby_places_tool(user_lat: float, user_lon: float, user_heading: float):
    """
    Factory that creates a search_nearby_places tool with user location bound.
    The tool returns places enriched with relative directions from the user.
    """

    class SearchNearbyInput(BaseModel):
        radius: float = Field(description="Search radius in meters. Use 50-70m for immediate surroundings, 100-200m for nearby blocks, 300-500m for broader neighborhood.")

    def search_nearby_places(radius: float) -> str:
        places = get_places(user_lat, user_lon, radius)

        # Enrich each place with relative direction and distance from user
        enriched_places = []
        for place in places:
            place_lat = place.get('latitude')
            place_lon = place.get('longitude')

            if place_lat and place_lon:
                direction, distance = calculate_relative_direction(
                    user_lat, user_lon, user_heading,
                    place_lat, place_lon
                )
                place['relative_direction'] = direction
                place['distance_meters'] = distance

            enriched_places.append(place)

        return json.dumps(enriched_places, indent=2)

    return StructuredTool.from_function(
        func=search_nearby_places,
        name="search_nearby_places",
        description="""Search for places around the user's current location within a specified radius.

Returns a list of places, each with:
- name, address, types
- relative_direction: "to your left", "directly ahead", "behind you", etc.
- distance_meters: how far from you in meters

Radius guidance:
- 50-70 meters: Immediate surroundings, buildings right next to you
- 100-200 meters: Nearby block, a short walk away
- 300-500 meters: Broader neighborhood context

You can call this multiple times with different radii to build a complete picture.""",
        args_schema=SearchNearbyInput
    )


def create_tavily_tool():
    """Create the Tavily search tool for researching places."""
    return TavilySearchResults(
        max_results=3,
        description="""Search the web for detailed information about a specific place or topic.
Use this to research interesting locations found nearby - their history,
significance, current activities, or any other relevant details.
Use the place name and address or neighborhood to construct your query.
Also use this for broader historical, cultural, or economic context about an area.

Good queries include:
- "History of [building name] [city]"
- "[Place name] architecture significance"
- "[Neighborhood name] economic history"
- "What is [place name] known for"
"""
    )


# --- Agent State ---

class TourGuideState(TypedDict):
    messages: Annotated[list, add_messages]


# --- System Prompt ---

SYSTEM_PROMPT = """You are an engaging virtual tour guide. Your job is to give the user a rich,
narrative tour of their surroundings based on their location and preferences.

## USER PREFERENCES
{user_preferences}

## USER LOCATION
- Coordinates: ({latitude}, {longitude})
- Currently facing: {heading_description} (heading: {heading}°)

## YOUR TOOLS

1. **search_nearby_places(radius)**: Searches around the user's location.
   - Returns places with `relative_direction` (e.g., "to your left", "directly ahead") and `distance_meters` already calculated.
   - Radius guidance:
     - 30-60m: Immediate surroundings, buildings right next to user
     - 100-200m: Nearby block
     - 300-500m: Broader neighborhood
   - Call multiple times with different radii if needed.

2. **tavily_search_results_json(query)**: Web search for researching places.
   - Use to look up history, significance, interesting facts about specific places.
   - Include place name + city/neighborhood for best results.

## YOUR WORKFLOW

1. First, call search_nearby_places with radius 30-60m for immediate surroundings
2. Optionally, call with larger radius (200-500m) for neighborhood context
3. Review the results. Based on user preferences, identify 2-4 most interesting places to highlight
4. Use tavily to research those places - get history, significance, interesting facts
5. Craft your narrative tour using the relative directions and distances from the search results

## OUTPUT FORMAT

Create an engaging narrative tour that:
- Uses the relative directions from the search results (e.g., "To your right, about 30 meters away...")
- Weaves together history, current significance, and interesting facts
- Connects individual places to broader narratives
- Flows naturally, as if walking with the user

## TONE
Be conversational and informative. Share fascinating details that make places come alive.
You're not listing places - you're telling the story of where they're standing.

Keep the final content relatively concise, and depth is preferreed over breadth. Focus on a few key highlights rather than overwhelming with too many details.
Remember that this is being read aloud, so clarity and flow are important, and it should feel like natural human narration.
That being said, do not search or spend time learning about many places. In your flow, just focus on two-three places and research them well. Try to be quick as the user is walking.
"""


# --- Graph Construction ---

def create_tour_guide_agent(user_lat: float, user_lon: float, user_heading: float):
    """
    Create and return the tour guide agent graph.

    Args:
        user_lat: User's latitude
        user_lon: User's longitude
        user_heading: User's facing direction in degrees (0=North)
    """
    # Create tools with user location bound
    search_tool = create_search_nearby_places_tool(user_lat, user_lon, user_heading)
    tavily_tool = create_tavily_tool()

    tools = [search_tool, tavily_tool]
    tool_node = ToolNode(tools)

    # Using OpenRouter to access Claude
    model = ChatOpenAI(
        model="anthropic/claude-sonnet-4.5",
        temperature=0.7,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    ).bind_tools(tools)

    def should_continue(state: TourGuideState):
        """Determine if we should continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def call_model(state: TourGuideState):
        """Call the model with the current state."""
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}

    # Build the graph
    graph = StateGraph(TourGuideState)

    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, ["tools", END])
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_tour(latitude: float, longitude: float, heading: float = 0.0, stream: bool = True):
    """
    Run the tour guide for a given location.

    Args:
        latitude: User's latitude
        longitude: User's longitude
        heading: User's facing direction in degrees (0=North, default)
        stream: If True, print live output for debugging

    Returns:
        The tour narrative
    """
    total_start = time.time()

    # Track timing for each step
    timings = {
        "agent_calls": [],  # Time spent in LLM calls
        "tool_calls": [],   # Time spent in tool execution
    }
    step_start = None
    current_node = None

    # Create agent with user location bound to tools
    agent = create_tour_guide_agent(latitude, longitude, heading)

    heading_desc = get_heading_description(heading)

    system_message = SYSTEM_PROMPT.format(
        user_preferences=user_preferences,
        latitude=latitude,
        longitude=longitude,
        heading=heading,
        heading_description=heading_desc
    )

    initial_state = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Give me an engaging tour of my surroundings!"}
        ]
    }

    if stream:
        print("\n[STREAMING MODE - Showing all events]\n")
        final_content = None

        for event in agent.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                # Record timing for previous step
                if step_start is not None and current_node is not None:
                    elapsed = time.time() - step_start
                    if current_node == "agent":
                        timings["agent_calls"].append(elapsed)
                    elif current_node == "tools":
                        timings["tool_calls"].append(elapsed)

                # Start timing new step
                step_start = time.time()
                current_node = node_name

                print(f"\n{'='*60}")
                print(f"NODE: {node_name}")
                print('='*60)

                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        # Check if it's an AI message with tool calls
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            print(f"\n[AI requesting tools]")
                            for tc in msg.tool_calls:
                                print(f"  Tool: {tc['name']}")
                                print(f"  Args: {tc['args']}")

                        # Check if it's a tool message (result)
                        elif hasattr(msg, 'type') and msg.type == "tool":
                            print(f"\n[Tool Result: {msg.name}]")
                            # Truncate long results for readability
                            content = str(msg.content)
                            if len(content) > 1000:
                                print(f"  {content[:1000]}...\n  [truncated, {len(content)} chars total]")
                            else:
                                print(f"  {content}")

                        # Regular AI message (final response)
                        elif hasattr(msg, 'content') and msg.content and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                            print(f"\n[AI Response]")
                            final_content = msg.content
                            print(msg.content[:500] + "..." if len(msg.content) > 500 else msg.content)

        # Record final step timing
        if step_start is not None and current_node is not None:
            elapsed = time.time() - step_start
            if current_node == "agent":
                timings["agent_calls"].append(elapsed)
            elif current_node == "tools":
                timings["tool_calls"].append(elapsed)

        total_time = time.time() - total_start

        # Print timing summary
        print(f"\n{'='*60}")
        print("TIMING SUMMARY")
        print('='*60)
        print(f"Total time: {total_time:.2f}s")
        print(f"\nLLM calls ({len(timings['agent_calls'])} total): {sum(timings['agent_calls']):.2f}s")
        for i, t in enumerate(timings['agent_calls'], 1):
            print(f"  Call {i}: {t:.2f}s")
        print(f"\nTool executions ({len(timings['tool_calls'])} total): {sum(timings['tool_calls']):.2f}s")
        for i, t in enumerate(timings['tool_calls'], 1):
            print(f"  Execution {i}: {t:.2f}s")

        return final_content
    else:
        result = agent.invoke(initial_state)
        total_time = time.time() - total_start
        print(f"\nTotal time: {total_time:.2f}s")
        return result["messages"][-1].content


# --- Main ---

if __name__ == "__main__":
    # Example: Near Union Square, San Francisco
    test_lat = 41.8303668
    test_lon = -71.4015215
    test_heading = 0  # Facing North

    print("Starting tour guide (OpenRouter)...\n")
    print("=" * 60)

    tour = run_tour(test_lat, test_lon, test_heading, stream=False)
    print(tour)
