import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

PREFERENCE_ENHANCEMENT_PROMPT = """You are helping to expand a user's brief tour preferences into a detailed, rich description that will guide an AI tour guide.

The user has provided a short description of what interests them on tours. Your job is to expand this into a comprehensive preference profile that covers:

1. **Historical interests**: What aspects of history fascinate them? (landmarks, hidden gems, how places connect to broader narratives, different time periods)
2. **Economic context**: Interest in how neighborhoods developed economically, what industries shaped areas, current economic activity
3. **Cultural and social aspects**: What makes places distinctive today (creative communities, local culture, unique characteristics)
4. **Architectural interests**: If mentioned, expand on their interest in building styles, design, etc.
5. **Overall tone**: What kind of stories and details would resonate with them?

**Important guidelines:**
- Maintain the user's core interests but expand them naturally
- Write in first person ("I'm fascinated by...", "I love...", "I want to understand...")
- Be conversational and natural, not robotic
- Keep it to 3-4 paragraphs maximum
- If they mention specific interests, emphasize those heavily
- If they're brief, infer complementary interests that naturally align

**User's brief preferences:**
{raw_preferences}

**Enhanced preference profile:**"""


def enhance_preferences(raw_preferences: str) -> str:
    """
    Use AI to enhance brief user preferences into a detailed profile.

    Args:
        raw_preferences: User's short description of their interests

    Returns:
        Enhanced, detailed preference description
    """
    model = ChatOpenAI(
        model="anthropic/claude-sonnet-4.5",
        temperature=0.7,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    prompt = PREFERENCE_ENHANCEMENT_PROMPT.format(raw_preferences=raw_preferences)
    response = model.invoke([{"role": "user", "content": prompt}])

    return response.content.strip()


def get_default_preferences() -> str:
    """
    Return default preferences if user hasn't set any yet.

    Returns:
        Default preference text
    """
    return """I'm interested in learning about the places around me. I enjoy discovering both
well-known landmarks and hidden gems with interesting stories. I appreciate understanding
the history of locations and how they connect to larger narratives. I'm curious about what
makes different neighborhoods unique - whether that's their architecture, culture, or the
communities that have shaped them over time."""


if __name__ == "__main__":
    # Test the enhancement function
    test_input = "I love history and cool architecture"
    print("Testing preference enhancement...")
    print(f"\nInput: {test_input}")
    print(f"\nEnhanced output:\n{enhance_preferences(test_input)}")
