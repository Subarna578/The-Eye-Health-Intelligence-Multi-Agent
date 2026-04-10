import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# Load environment variables from .env file
load_dotenv()

# Configure OpenRouter model
# IMPORTANT: API key is loaded from .env file (see .env.example)
model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000
)


#Create a tool


# Create specialized logistics planning agent
logistics_agent = create_agent(
    model=model,
    system_prompt="""You are a travel logistics expert. You handle practical travel planning:
    - Calculate distances between locations and travel times
    - Estimate costs for transportation, accommodation, and activities
    - Optimize routes and suggest efficient itineraries
    - Consider time zones, weather, and practical constraints
    Always provide short, clear, practical logistics information."""
)

#Create specialized recommendation agent
recommendation_agent = create_agent(
    model=model,
    system_prompt="""You are a travel recommendations expert. You suggest experiences and activities:
    - Recommend top attractions , landmarks, and must-see sights
    - Suggest restaurants, local cuisine, and dining experiences
    - Recommend cultural activities, events, and unique local experiences
    - Consider user budget and travel style
    - Provide insights about local customs, best time to visit, and hidden gems
    Always provide brief, engaging, personalized recommendations."""
)


#Create the tool


@tool
def plan_logistics_agent(trip_request: str) -> str:
    """
    Plan travel logistics including distances, times, costs, and routes.
    Use this to calculate practical travel information and optimize itineraries.
    
    Args:
        trip_request: Trip details (e.g., "3 days in Paris, budget $1500, from London")
    
    Returns:
        Logistics information: distances, travel times, costs, and route suggestions
    """
    response = logistics_agent.invoke({"messages": [HumanMessage(f"Plan logistics for this trip: {trip_request}")]})
    return response["messages"][-1].content

@tool
def get_recommendations_agent(trip_details: str) -> str:
    """
    Get travel recommendations for attractions, restaurants, and activities.
    Use this to suggest what to see, do, and eat at the destination.
    
    Args:
        trip_details: Destination and trip information (e.g., "3 days in Paris, interested in art and food")
    
    Returns:
        Recommendations for attractions, restaurants, activities, and cultural insights
    """
    response = recommendation_agent.invoke({"messages": [HumanMessage(f"Provide recommendations for: {trip_details}")]})
    return response["messages"][-1].content


# Create Orchestrator Agent : the agent that coordinates everything. 


orchestrator = create_agent(
    model = model,
    system_prompt="""You are a travel planning coordinator. 
    When planning trips, use both specialists:
    1. Use plan_logistics_agent to calculate practical details: distances, times, costs, and routes
    2. Use get_recommendations_agent to suggest attractions, restaurants, and activities
    Always combine both the practical logistics and exciting recommendations in your final response.""",
    tools=[plan_logistics_agent, get_recommendations_agent]
)

# Test the orchestrator with a sample trip request
if __name__ == "__main__":
    trip_request = "Plan a 3-day trip to Paris for a budget of $1500, departing from London. I'm interested in art and food."
    response = orchestrator.invoke({"messages": [HumanMessage(trip_request)]})
    print(response["messages"][-1].content)

