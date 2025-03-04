import os
from swarm_models import LiteLLM # Uncomment the LiteLLM in swarms_model
from google import genai

api_key = os.getenv("GEMINI_API_KEY")
llm = LiteLLM(model_name="gemini/gemini-2.0-flash")
client = genai.Client(api_key=api_key)

