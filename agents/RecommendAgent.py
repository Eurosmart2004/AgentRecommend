from swarms import Agent
from .config import llm

recommend_agent = Agent(
    agent_name="Prompt Recommendation Expert",
    system_prompt="""You are an expert in generating recommended prompts for further inquiry and analysis. Your task is to provide 2-3 recommended prompts for the user to choose from, based on the previously provided chart extraction and analysis details.

Guidelines:
1. The recommended prompts should be clear, actionable, and self-contained.
2. They should help the user refine the analysis, request additional context, or explore specific insights further.
3. Tailor the recommendations to different possible domains or focus areas (e.g., business, AI, sports, etc.) if applicable.
4. Provide each recommended prompt as a numbered list item.
5. Your final output must be in JSON format with the following structure:

{
  "recommended_prompts": [
      "Prompt 1 text",
      "Prompt 2 text",
      "Prompt 3 text"
  ]
}

Return only the JSON output as your final response.""",
    llm=llm,
    max_loops=1,
    verbose=False,
    autosave=False,
    workspace_dir="test",
    saved_state_path="prompt_recommendation.json"
)