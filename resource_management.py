import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from trace_viewer import get_trace_info, print_results

load_dotenv()

model_id = "gpt-4o-mini"

model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model=model_id,
    temperature=0.7,
    max_tokens=1000
)

print(f"✓ Model configured: {model_id}")

#Initialize langfuse and helper functions

"""
    How Langfuse works with Langchain:
    The integration combines two mechanisms:
    1. @observe() decorator : wraps a function to automatically create a langfuse trace on each call. All langfuse operations inside the decorated function are nested under that trace.
    2. CallBackHandler() - created inside the @observe() function, it automatically attaches to the current trace and captures LangChain-specific metrics ( token, costs, latency)
    3. Session tracking - Multiple calls can be grouped under the same session_id using langfuse_client.update_current_trace(session_id=...). This lets you group all calls from a single run together.
    4. Unique session IDs - Generated with ulid in the format {TEAM_NAME}-{ULID} for easy identification 

    What gets tracked automatically
    The CallBackHandler captures:

    1. Inputs and outputs - All messages sent to and received from the model
    2. Token usage - Input, output, and cache tokens ( when available)
    3. Costs - automatically calculated based on model pricing
    4. Latency - time taken for each operation
    5. Metadata - model parameters, temperature, etc.
"""

import ulid
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

# Initialize Langfuse client
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
)

def generate_session_id():
    """Generate a unique session ID using TEAM_NAME and ULID."""
    return f"{os.getenv('TEAM_NAME', 'tutorial')}-{ulid.new().str}"

def invoke_langchain(model, prompt, langfuse_handler):
    """Invoke LangChain with the given prompt and Langfuse handler."""
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages, config={"callbacks": [langfuse_handler]})
    return response.content

@observe()
def run_llm_call(session_id, model, prompt):
    """Run a single LangChain invocation and track it in Langfuse."""
    # Update trace with session_id
    langfuse_client.update_current_trace(session_id=session_id)

    # Create Langfuse callback handler for automatic generation tracking
    # The handler will attach to the current trace created by @observe()
    langfuse_handler = CallbackHandler()

    # Invoke LangChain with Langfuse handler to track tokens and costs
    response = invoke_langchain(model, prompt, langfuse_handler)

    return response

print("✓ Langfuse initialized successfully")
print(f"✓ Public key: {os.getenv('LANGFUSE_PUBLIC_KEY', 'Not set')[:20]}...")
print("✓ Helper functions ready: generate_session_id(), invoke_langchain(), run_llm_call()")


questions = [
    "What is machine learning?",
    "Explain neural networks briefly",
    "What is the difference between AI and ML?"
]

session_id = generate_session_id()

print(f"Session ID: {session_id}")
print(f"Making {len(questions)} agent calls with Langfuse tracking...")

for i, question in enumerate(questions, 1):
    response = run_llm_call(session_id, model, question)
    print(f"Call {i}: {question[:40]}...")
    print(f"   Response: {response[:80]}...")

langfuse_client.flush()

info = get_trace_info(session_id= session_id, langfuse_client= langfuse_client)

print_results(info)

