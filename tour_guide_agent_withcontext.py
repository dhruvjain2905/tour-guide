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
from location_claude import get_places


# --- Structured Output Models ---

class PlaceCoordinate(BaseModel):
    """A place mentioned in the tour with its coordinates."""
    name: str = Field(description="Name of the place")
    latitude: float = Field(description="Latitude coordinate")
    longitude: float = Field(description="Longitude coordinate")
    relative_direction: str | None = Field(default=None, description="Direction from user (e.g., 'to your left', 'directly ahead')")
    distance_meters: float | None = Field(default=None, description="Distance from user in meters")


class TourOutput(BaseModel):
    """Structured output containing the tour narrative and places discussed."""
    narrative: str = Field(description="The engaging tour narrative text")
    places: list[PlaceCoordinate] = Field(description="List of places discussed in the tour with their coordinates")


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


def clean_text_output(text: str) -> str:
    """
    Clean text output to remove markdown formatting and special characters.
    Returns plain text with only sentences and proper spacing.
    """
    import re

    # Remove markdown bold/italic (**, *, __, _)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)

    # Remove markdown headers (#, ##, ###, etc.)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

    # Remove bullet points (-, *, •) at the start of lines
    text = re.sub(r'^\s*[-*•]\s+', '    ', text, flags=re.MULTILINE)

    # Remove numbered lists (1., 2., etc.) at the start of lines
    text = re.sub(r'^\s*\d+\.\s+', '    ', text, flags=re.MULTILINE)

    # Remove excessive newlines (more than 2 in a row)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Clean up any remaining isolated special characters
    text = re.sub(r'(?<!\S)[*#]+(?!\S)', '', text)

    return text.strip()


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


# --- System Prompts ---

SYSTEM_PROMPT_START = """You are an engaging virtual tour guide. Your job is to give the user a rich,
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

CRITICAL FORMATTING RULES:
- Output ONLY plain text sentences and paragraphs
- Do NOT use asterisks, stars, bullet points, or any markdown formatting
- Do NOT use special characters like *, **, -, #, or numbered lists
- The output will be read aloud, so it must be clean natural speech

## TONE
Be conversational and informative. Share fascinating details that make places come alive.
You're not listing places - you're telling the story of where they're standing.

Keep the final content relatively concise, and depth is preferred over breadth. Focus on a few key highlights rather than overwhelming with too many details.
Remember that this is being read aloud, so clarity and flow are important, and it should feel like natural human narration.
That being said, do not search or spend time learning about many places. In your flow, just focus on two-three places and research them well. Try to be quick as the user is walking.

MOST IMORTANTLY: Do not be too verbose of have long output. Keep it short, deep, and sweet, and interesting, as the user is walking.
Act like a normal human tour guide who is walking with the user, and not an encyclopedia that is dumping information.
"""

SYSTEM_PROMPT_CONTINUE = """You are an engaging virtual tour guide. You are CONTINUING an ongoing walking tour.
The user has moved to a new location, and you must seamlessly continue the narrative.

## USER PREFERENCES
{user_preferences}

## CURRENT LOCATION
- Coordinates: ({latitude}, {longitude})
- Currently facing: {heading_description} (heading: {heading}°)

## PREVIOUS TOUR SUMMARY
This is what has been covered so far in the tour:
{previous_summary}

## CRITICAL CONTINUATION GUIDELINES
- DO NOT repeat facts or places already mentioned in the previous summary
- Vary your narrative angles and style - if previous sections focused on history, try architecture or local culture
- Create natural transitions that acknowledge the journey (e.g., "As we continue...", "Moving along...", "Now that we've seen...")
- Build on themes from earlier if relevant, but don't rehash them
- The tour should feel like one continuous experience, not separate segments

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
3. Review the results - SKIP any places already covered in the previous summary
4. Based on user preferences, identify 2-3 NEW interesting places to highlight
5. Use tavily to research those places - get history, significance, interesting facts
6. Craft your narrative as a CONTINUATION of the tour

## OUTPUT FORMAT

Create an engaging narrative tour segment that:
- Flows naturally from where the previous segment left off
- Uses the relative directions from the search results
- Introduces fresh content and perspectives
- Feels like the next chapter of the same story

CRITICAL FORMATTING RULES:
- Output ONLY plain text sentences and paragraphs
- Do NOT use asterisks, stars, bullet points, or any markdown formatting
- Do NOT use special characters like *, **, -, #, or numbered lists
- The output will be read aloud, so it must be clean natural speech

## TONE
Be conversational and informative. Maintain consistency with the tour's overall feel while keeping it fresh.
Remember this is being read aloud - clarity and natural flow are essential.

MOST IMORTANTLY: Do not be too verbose of have long output. Keep it short, deep, and sweet, and interesting, as the user is walking.
Act like a normal human tour guide who is walking with the user, and not an encyclopedia that is dumping information.
"""

SUMMARY_PROMPT = """You are summarizing a tour guide segment for continuity purposes.

Given the following tour narrative, create a concise summary that captures:
1. The key places/landmarks mentioned (names and brief descriptions)
2. Main historical facts or stories shared
3. Any themes or narrative threads that were developed
4. The general area/neighborhood covered

Keep the summary factual and concise (3-5 sentences). This will be used to help the next tour segment avoid repetition and maintain continuity.

CRITICAL FORMATTING RULES:
- Output ONLY plain text sentences
- Do NOT use asterisks, stars, bullet points, or any markdown formatting
- Do NOT use special characters like *, **, -, #, or numbered lists
- Write flowing prose sentences only

TOUR NARRATIVE:
{tour_text}

SUMMARY:"""


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


def generate_summary(tour_text: str) -> str:
    """
    Generate a concise summary of a tour segment for continuity.

    Args:
        tour_text: The tour narrative to summarize

    Returns:
        A concise summary for use in the next segment
    """
    model = ChatOpenAI(
        model="anthropic/claude-sonnet-4.5",
        temperature=0.3,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    prompt = SUMMARY_PROMPT.format(tour_text=tour_text)
    response = model.invoke([{"role": "user", "content": prompt}])
    return clean_text_output(response.content)


def format_tour_output(messages: list) -> TourOutput:
    """
    Take the agent conversation and produce structured output using with_structured_output.

    Args:
        messages: The full conversation history from the agent

    Returns:
        TourOutput with narrative and places
    """
    model = ChatOpenAI(
        model="anthropic/claude-sonnet-4.5",
        temperature=0.7,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    ).with_structured_output(TourOutput)

    # Add instruction to format as structured output
    format_instruction = {
        "role": "user",
        "content": "Now provide your tour response as structured output with the narrative and the places you discussed (with their coordinates from the search results)."
    }

    return model.invoke(messages + [format_instruction])


def run_tour(
    latitude: float,
    longitude: float,
    heading: float = 0.0,
    previous_summary: str | None = None,
    stream: bool = True,
    user_preferences: str | None = None
) -> dict:
    """
    Run the tour guide for a given location.

    Args:
        latitude: User's latitude
        longitude: User's longitude
        heading: User's facing direction in degrees (0=North, default)
        previous_summary: Summary of the tour so far (None if starting fresh)
        stream: If True, print live output for debugging
        user_preferences: User's preferences text (if None, uses defaults)

    Returns:
        dict with 'tour' (TourOutput with narrative and places), and 'summary' (for next segment)
    """
    # Use default preferences if none provided
    if user_preferences is None:
        from location_claude import user_preferences as default_prefs
        user_preferences = default_prefs

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

    # Choose system prompt based on whether this is a continuation
    if previous_summary:
        system_message = SYSTEM_PROMPT_CONTINUE.format(
            user_preferences=user_preferences,
            latitude=latitude,
            longitude=longitude,
            heading=heading,
            heading_description=heading_desc,
            previous_summary=previous_summary
        )
    else:
        system_message = SYSTEM_PROMPT_START.format(
            user_preferences=user_preferences,
            latitude=latitude,
            longitude=longitude,
            heading=heading,
            heading_description=heading_desc
        )

    # Adjust user message based on whether continuing or starting
    if previous_summary:
        user_message = "Continue the tour from my new location!"
    else:
        user_message = "Give me an engaging tour of my surroundings!"

    initial_state = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    }

    if stream:
        print("\n[STREAMING MODE - Showing all events]\n")
        all_messages = list(initial_state["messages"])  # Track all messages

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
                        all_messages.append(msg)  # Capture message

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
                            print(f"\n[AI Response Preview]")
                            print(str(msg.content)[:500] + "..." if len(str(msg.content)) > 500 else msg.content)

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

        # Get structured output using with_structured_output
        print(f"\n{'='*60}")
        print("GENERATING STRUCTURED OUTPUT...")
        print('='*60)
        struct_start = time.time()
        tour_output = format_tour_output(all_messages)
        print(f"Structured output generated in {time.time() - struct_start:.2f}s")
        print(f"Found {len(tour_output.places)} places:")
        for place in tour_output.places:
            print(f"  - {place.name}: ({place.latitude}, {place.longitude})")

        # Generate summary for next segment
        if tour_output and tour_output.narrative:
            print(f"\n{'='*60}")
            print("GENERATING SUMMARY...")
            print('='*60)
            summary_start = time.time()
            # Clean the narrative for summary generation
            cleaned_narrative = clean_text_output(tour_output.narrative)
            summary = generate_summary(cleaned_narrative)
            print(f"Summary generated in {time.time() - summary_start:.2f}s")
            print(f"Summary: {summary}")
        else:
            summary = None

        return {"tour": tour_output, "summary": summary}
    else:
        result = agent.invoke(initial_state)
        total_time = time.time() - total_start
        print(f"\nAgent time: {total_time:.2f}s")

        # Get structured output using with_structured_output
        tour_output = format_tour_output(result["messages"])

        # Generate summary from the narrative
        cleaned_narrative = clean_text_output(tour_output.narrative)
        summary = generate_summary(cleaned_narrative)

        return {"tour": tour_output, "summary": summary}


# --- Main ---

if __name__ == "__main__":
    # Example: Starting location
    test_lat = 41.8303668
    test_lon = -71.4015215
    test_heading = 0  # Facing North

    print("Starting tour guide (OpenRouter)...\n")
    print("=" * 60)

    # First segment - no previous summary
    result = run_tour(test_lat, test_lon, test_heading, previous_summary=None, stream=True)
    print("\n" + "=" * 60)
    print("TOUR OUTPUT (STRUCTURED):")
    print("=" * 60)
    if result["tour"]:
        print("\n--- NARRATIVE ---")
        print(clean_text_output(result["tour"].narrative))
        print("\n--- PLACES WITH COORDINATES (JSON) ---")
        places_json = [
            {
                "name": p.name,
                "latitude": p.latitude,
                "longitude": p.longitude,
                "relative_direction": p.relative_direction,
                "distance_meters": p.distance_meters
            }
            for p in result["tour"].places
        ]
        print(json.dumps(places_json, indent=2))
    print("\n" + "=" * 60)
    print("SUMMARY FOR NEXT SEGMENT:")
    print("=" * 60)
    print(result["summary"])


    result = run_tour(test_lat, test_lon, test_heading, previous_summary=result["summary"], stream=True)
    print("\n" + "=" * 60)
    print("TOUR OUTPUT (STRUCTURED):")
    print("=" * 60)
    if result["tour"]:
        print("\n--- NARRATIVE ---")
        print(clean_text_output(result["tour"].narrative))
        print("\n--- PLACES WITH COORDINATES (JSON) ---")
        places_json = [
            {
                "name": p.name,
                "latitude": p.latitude,
                "longitude": p.longitude,
                "relative_direction": p.relative_direction,
                "distance_meters": p.distance_meters
            }
            for p in result["tour"].places
        ]
        print(json.dumps(places_json, indent=2))
    print("\n" + "=" * 60)
    print("SUMMARY FOR NEXT SEGMENT:")
    print("=" * 60)
    print(result["summary"])

    # # Example of continuing the tour at a new location:
    # next_result = run_tour(
    #      latitude=41.8317,
    #      longitude=-71.4020,
    #      heading=90,  # Now facing East
    #      previous_summary=result["summary"],
    #      stream=True
    #  )
    #
    # print("\n" + "=" * 60)
    # print("TOUR OUTPUT (STRUCTURED):")
    # print("=" * 60)
    # if next_result["tour"]:
    #     print("\n--- NARRATIVE ---")
    #     print(clean_text_output(next_result["tour"].narrative))
    #     print("\n--- PLACES WITH COORDINATES (JSON) ---")
    #     places_json = [
    #         {
    #             "name": p.name,
    #             "latitude": p.latitude,
    #             "longitude": p.longitude,
    #             "relative_direction": p.relative_direction,
    #             "distance_meters": p.distance_meters
    #         }
    #         for p in next_result["tour"].places
    #     ]
    #     print(json.dumps(places_json, indent=2))
    # print("\n" + "=" * 60)
    # print("SUMMARY FOR NEXT SEGMENT:")
    # print("=" * 60)
    # print(next_result["summary"])