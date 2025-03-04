from .AnalyzeAgent import analyze_agent
from .AnswerQueryAgent import answer_query_agent
from .utils import *
from .config import api_key


agent_router = MultiAgentRouter(
    name="MultiAgentRouter",
    description="""Routes user queries to the appropriate agent.
        If the query is about analyzing a chart image, route it to analyze_agent.
        Otherwise, route it to answer_query_agent.""",
    agents=[analyze_agent, answer_query_agent],
    model="gemini/gemini-2.0-flash",
    api_key=api_key,
)