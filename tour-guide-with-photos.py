import os
import math
import json
import time
from typing import Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool, StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from location_with_photos import get_places, user_preferences


load_dotenv()  # Load environment variables from .env file


# --- Structured Output Models ---

class TextSegment(BaseModel):
    """A text segment of the tour narrative."""
    type: Literal["text"] = "text"
    content: str = Field(description="The narrative text content")


class PhotoSegment(BaseModel):
    """A photo segment to display alongside the narrative."""
    type: Literal["photo"] = "photo"
    place_name: str = Field(description="Name of the place this photo shows")
    photo_url: str = Field(description="The EXACT photo URL from the search results - copy it precisely, character for character")


class TourOutput(BaseModel):
    """Structured tour output with interleaved text and photos."""
    segments: list[TextSegment | PhotoSegment] = Field(
        description="List of text and photo segments that make up the tour. Interleave photos naturally with the narrative text."
    )


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

def create_search_nearby_places_tool(user_lat: float, user_lon: float, user_heading: float):
    """
    Factory that creates a search_nearby_places tool with user location bound.
    The tool returns places enriched with relative directions and photo URLs.
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
- photo_url: URL to a photo of the place (use this EXACTLY when including photos in your output)

IMPORTANT: When you want to include a photo in your tour output, you MUST copy the photo_url
EXACTLY as it appears in these results. Do not modify, truncate, or paraphrase the URL.

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
   - Returns places with `relative_direction`, `distance_meters`, and `photo_url`
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
4. **IMPORTANT**: Note which places have photo_url values - you will use these exact URLs in your output
5. Use tavily to research those places - get history, significance, interesting facts
6. Craft your narrative tour using the structured output format

## OUTPUT FORMAT - CRITICAL

You MUST return a JSON object with this exact structure:
{{
  "segments": [
    {{"type": "text", "content": "Your narrative text here..."}},
    {{"type": "photo", "place_name": "Place Name", "photo_url": "EXACT_URL_FROM_SEARCH_RESULTS"}},
    {{"type": "text", "content": "More narrative continuing after the photo..."}},
    ...
  ]
}}

CRITICAL RULES FOR PHOTOS:
- The photo_url MUST be copied EXACTLY from the search_nearby_places results - character for character
- Do NOT modify, shorten, or paraphrase the URLs
- Only include photos for places that have a photo_url in the search results (some may be null)
- Place photos naturally in the narrative - introduce a place with text, show its photo, then continue
- Match the photo to the place you're currently discussing

## EXAMPLE OUTPUT
{{
  "segments": [
    {{"type": "text", "content": "Welcome! To your right, about 30 meters away, stands the historic Trinity Church."}},
    {{"type": "photo", "place_name": "Trinity Church", "photo_url": "https://places.googleapis.com/v1/places/ChIJ.../photos/AUc.../media?key=...&maxHeightPx=400&maxWidthPx=400"}},
    {{"type": "text", "content": "Built in 1726, this Georgian masterpiece has witnessed centuries of American history..."}}
  ]
}}

## TONE
Be conversational and informative. Share fascinating details that make places come alive.
You're not listing places - you're telling the story of where they're standing.

Keep the final content relatively concise, and depth is preferred over breadth.
Remember that the text is being read aloud, so clarity and flow are important.
Focus on 2-3 places and research them well. Try to be quick as the user is walking.
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
- Vary your narrative angles and style
- Create natural transitions that acknowledge the journey
- Build on themes from earlier if relevant, but don't rehash them

## YOUR TOOLS

1. **search_nearby_places(radius)**: Searches around the user's location.
   - Returns places with `relative_direction`, `distance_meters`, and `photo_url`
   - IMPORTANT: Note the photo_url for each place - you'll need these exact URLs

2. **tavily_search_results_json(query)**: Web search for researching places.

## OUTPUT FORMAT - CRITICAL

You MUST return a JSON object with this exact structure:
{{
  "segments": [
    {{"type": "text", "content": "Your narrative text here..."}},
    {{"type": "photo", "place_name": "Place Name", "photo_url": "EXACT_URL_FROM_SEARCH_RESULTS"}},
    {{"type": "text", "content": "More narrative continuing after the photo..."}},
    ...
  ]
}}

CRITICAL RULES FOR PHOTOS:
- The photo_url MUST be copied EXACTLY from the search_nearby_places results - character for character
- Do NOT modify, shorten, or paraphrase the URLs
- Only include photos for places that have a photo_url (not null)
- Place photos naturally in the narrative flow

## TONE
Be conversational and informative. Maintain consistency with the tour's overall feel while keeping it fresh.
Remember the text is being read aloud - clarity and natural flow are essential.
"""

SUMMARY_PROMPT = """You are summarizing a tour guide segment for continuity purposes.

Given the following tour narrative, create a concise summary that captures:
1. The key places/landmarks mentioned (names and brief descriptions)
2. Main historical facts or stories shared
3. Any themes or narrative threads that were developed
4. The general area/neighborhood covered

Keep the summary factual and concise (3-5 sentences). This will be used to help the next tour segment avoid repetition and maintain continuity.

TOUR NARRATIVE:
{tour_text}

SUMMARY:"""


# --- Graph Construction ---

def create_tour_guide_agent(user_lat: float, user_lon: float, user_heading: float):
    """
    Create and return the tour guide agent graph.
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


def extract_tour_text(tour_output: TourOutput) -> str:
    """Extract just the text content from a structured tour output for summarization."""
    text_parts = []
    for segment in tour_output.segments:
        if segment.type == "text":
            text_parts.append(segment.content)
    return " ".join(text_parts)


def generate_summary(tour_text: str) -> str:
    """Generate a concise summary of a tour segment for continuity."""
    model = ChatOpenAI(
        model="anthropic/claude-sonnet-4.5",
        temperature=0.3,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    prompt = SUMMARY_PROMPT.format(tour_text=tour_text)
    response = model.invoke([{"role": "user", "content": prompt}])
    return response.content


def parse_tour_output(content: str) -> TourOutput:
    """Parse the LLM's response into a structured TourOutput."""
    try:
        # Try to find JSON in the response
        # Sometimes the LLM might wrap it in markdown code blocks
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0]

        data = json.loads(json_str)
        return TourOutput(**data)
    except (json.JSONDecodeError, ValueError) as e:
        # Fallback: treat entire content as a single text segment
        print(f"Warning: Could not parse structured output, falling back to text: {e}")
        return TourOutput(segments=[TextSegment(content=content)])


def run_tour(
    latitude: float,
    longitude: float,
    heading: float = 0.0,
    previous_summary: str | None = None,
    stream: bool = True
) -> dict:
    """
    Run the tour guide for a given location.

    Returns:
        dict with 'tour' (TourOutput object), 'tour_json' (serialized), and 'summary' (for next segment)
    """
    total_start = time.time()

    timings = {
        "agent_calls": [],
        "tool_calls": [],
    }
    step_start = None
    current_node = None

    agent = create_tour_guide_agent(latitude, longitude, heading)
    heading_desc = get_heading_description(heading)

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

    user_message = "Continue the tour from my new location!" if previous_summary else "Give me an engaging tour of my surroundings!"

    initial_state = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    }

    if stream:
        print("\n[STREAMING MODE - Showing all events]\n")
        final_content = None

        for event in agent.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                if step_start is not None and current_node is not None:
                    elapsed = time.time() - step_start
                    if current_node == "agent":
                        timings["agent_calls"].append(elapsed)
                    elif current_node == "tools":
                        timings["tool_calls"].append(elapsed)

                step_start = time.time()
                current_node = node_name

                print(f"\n{'='*60}")
                print(f"NODE: {node_name}")
                print('='*60)

                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            print(f"\n[AI requesting tools]")
                            for tc in msg.tool_calls:
                                print(f"  Tool: {tc['name']}")
                                print(f"  Args: {tc['args']}")

                        elif hasattr(msg, 'type') and msg.type == "tool":
                            print(f"\n[Tool Result: {msg.name}]")
                            content = str(msg.content)
                            if len(content) > 1000:
                                print(f"  {content[:1000]}...\n  [truncated, {len(content)} chars total]")
                            else:
                                print(f"  {content}")

                        elif hasattr(msg, 'content') and msg.content and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                            print(f"\n[AI Response]")
                            final_content = msg.content
                            print(msg.content[:500] + "..." if len(msg.content) > 500 else msg.content)

        if step_start is not None and current_node is not None:
            elapsed = time.time() - step_start
            if current_node == "agent":
                timings["agent_calls"].append(elapsed)
            elif current_node == "tools":
                timings["tool_calls"].append(elapsed)

        total_time = time.time() - total_start

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

        if final_content:
            print(f"\n{'='*60}")
            print("PARSING STRUCTURED OUTPUT...")
            print('='*60)
            tour_output = parse_tour_output(final_content)

            print(f"\nParsed {len(tour_output.segments)} segments:")
            for i, seg in enumerate(tour_output.segments):
                if seg.type == "text":
                    preview = seg.content[:100] + "..." if len(seg.content) > 100 else seg.content
                    print(f"  {i+1}. [TEXT] {preview}")
                else:
                    print(f"  {i+1}. [PHOTO] {seg.place_name}")

            print(f"\n{'='*60}")
            print("GENERATING SUMMARY...")
            print('='*60)
            summary_start = time.time()
            tour_text = extract_tour_text(tour_output)
            summary = generate_summary(tour_text)
            print(f"Summary generated in {time.time() - summary_start:.2f}s")
            print(f"Summary: {summary}")
        else:
            tour_output = None
            summary = None

        return {
            "tour": tour_output,
            "tour_json": tour_output.model_dump() if tour_output else None,
            "summary": summary
        }
    else:
        result = agent.invoke(initial_state)
        final_content = result["messages"][-1].content
        total_time = time.time() - total_start
        print(f"\nTotal time: {total_time:.2f}s")

        tour_output = parse_tour_output(final_content)
        tour_text = extract_tour_text(tour_output)
        summary = generate_summary(tour_text)

        return {
            "tour": tour_output,
            "tour_json": tour_output.model_dump(),
            "summary": summary
        }


# --- Main ---

if __name__ == "__main__":
    test_lat = 42.4078751
    test_lon = -71.1173936
    test_heading = 0  # Facing North

    print("Starting tour guide with photos (OpenRouter)...\n")
    print("=" * 60)

    result = run_tour(test_lat, test_lon, test_heading, previous_summary=None, stream=True)

    print("\n" + "=" * 60)
    print("STRUCTURED TOUR OUTPUT:")
    print("=" * 60)
    if result["tour"]:
        print(json.dumps(result["tour_json"], indent=2))

    print("\n" + "=" * 60)
    print("SUMMARY FOR NEXT SEGMENT:")
    print("=" * 60)
    print(result["summary"])
