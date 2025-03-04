import os
import uuid
from datetime import datetime
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel, Field
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.output_type import OutputType
from swarms.utils.any_to_str import any_to_str

from .function_caller_model import LiteLLMFunctionCaller


class AgentResponse(BaseModel):
    """Response from the boss agent indicating which agent should handle the task"""

    selected_agent: str = Field(
        description="Name of the agent selected to handle the task"
    )
    reasoning: str = Field(
        description="Explanation for why this agent was selected"
    )
    modified_task: Optional[str] = Field(
        None, description="Optional modified version of the task"
    )


class MultiAgentRouter:
    """
    Routes tasks to appropriate agents based on their capabilities.

    This class is responsible for managing a pool of agents and routing incoming tasks to the most suitable agent. It uses a boss agent to analyze the task and select the best agent for the job. The boss agent's decision is based on the capabilities and descriptions of the available agents.

    Attributes:
        name (str): The name of the router.
        description (str): A description of the router's purpose.
        agents (dict): A dictionary of agents, where the key is the agent's name and the value is the agent object.
        api_key (str): The API key for OpenAI.
        output_type (str): The type of output expected from the agents. Can be either "json" or "string".
        execute_task (bool): A flag indicating whether the task should be executed by the selected agent.
        boss_system_prompt (str): A system prompt for the boss agent that includes information about all available agents.
        function_caller (OpenAIFunctionCaller): An instance of OpenAIFunctionCaller for calling the boss agent.
    """

    def __init__(
        self,
        name: str = "swarm-router",
        description: str = "Routes tasks to specialized agents based on their capabilities",
        agents: List[Agent] = [],
        model: str = "gemini/gemini-2.0-flash",
        api_key: str = os.getenv("GEMINI_API_KEY"),
        temperature: float = 0.1,
        shared_memory_system: callable = None,
        output_type: OutputType = "dict",
        execute_task: bool = True,
    ):
        """
        Initializes the MultiAgentRouter with a list of agents and configuration options.

        Args:
            name (str, optional): The name of the router. Defaults to "swarm-router".
            description (str, optional): A description of the router's purpose. Defaults to "Routes tasks to specialized agents based on their capabilities".
            agents (List[Agent], optional): A list of agents to be managed by the router. Defaults to an empty list.
            model (str, optional): The model to use for the boss agent. Defaults to "gpt-4-0125-preview".
            temperature (float, optional): The temperature for the boss agent's model. Defaults to 0.1.
            output_type (Literal["json", "string"], optional): The type of output expected from the agents. Defaults to "json".
            execute_task (bool, optional): A flag indicating whether the task should be executed by the selected agent. Defaults to True.
        """
        self.name = name
        self.description = description
        self.shared_memory_system = shared_memory_system
        self.output_type = output_type
        self.execute_task = execute_task
        self.model = model
        self.temperature = temperature

        # Initialize Agents
        self.agents = {agent.name: agent for agent in agents}
        self.conversation = Conversation()

        self.api_key = api_key
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided")

        self.boss_system_prompt = self._create_boss_system_prompt()

        self.function_caller = LiteLLMFunctionCaller(
            model_name=self.model,
            system_prompt=self.boss_system_prompt,
            base_model=AgentResponse,
            api_key=self.api_key,
            temperature=self.temperature,
        )

    def __repr__(self):
        return f"MultiAgentRouter(name={self.name}, agents={list(self.agents.keys())})"

    def query_ragent(self, task: str) -> str:
        """Query the ResearchAgent"""
        return self.shared_memory_system.query(task)

    def _create_boss_system_prompt(self) -> str:
        """
        Creates a system prompt for the boss agent that includes information about all available agents.

        Returns:
            str: The system prompt for the boss agent.
        """
        agent_descriptions = "\n".join(
            [
                f"- {name}: {agent.description}"
                for name, agent in self.agents.items()
            ]
        )

        return f"""You are a task-routing boss agent responsible for assigning tasks to the most suitable specialized agent.  

### Available Agents:
{agent_descriptions}  

### Your Responsibilities:
1. **Analyze** the given task.  
2. **Identify** the most appropriate agent based on their expertise.  
3. **Explain** your selection with clear reasoning.  
4. **Enhance** the task while preserving its original intent, making it clearer and more actionable for the selected agent.  

### Response Format (JSON):
You must respond with a valid JSON object containing:  
- **selected_agent**: The name of the chosen agent (must be one of the available agents).  
- **reasoning**: A brief explanation of why this agent was selected.  
- **modified_task** *(optional)*: A refined version of the task that:
  - Retains **all key details** from the original task.  
  - Clarifies vague elements if necessary.  
  - Expands on missing details that could help the agent execute the task better.  

⚠️ Always select exactly **one** agent that best matches the task requirements."""

    def route_task(self, task: str) -> dict:
        """
        Routes a task to the appropriate agent and returns their response.

        Args:
            task (str): The task to be routed.

        Returns:
            dict: A dictionary containing the routing result, including the selected agent, reasoning, and response.
        """
        try:
            self.conversation.add(role="user", content=task)
            start_time = datetime.now()

            # Get boss decision using function calling
            boss_response: AgentResponse = self.function_caller.run(task)

            boss_response_str = any_to_str(boss_response)

            self.conversation.add(
                role="assistant", content=boss_response_str
            )

            # Validate that the selected agent exists
            if boss_response.selected_agent not in self.agents:
                raise ValueError(
                    f"Boss selected unknown agent: {boss_response.selected_agent}"
                )

            # Get the selected agent
            selected_agent = self.agents[boss_response.selected_agent]

            # Use the modified task if provided, otherwise use original task
            # final_task = boss_response.modified_task or task
            final_task = task

            # Execute the task with the selected agent if enabled
            execution_start = datetime.now()
            agent_response = None
            execution_time = 0

            if self.execute_task:
                # Use the agent's run method directly
                agent_response = selected_agent.run(final_task)
                self.conversation.add(
                    role=selected_agent.name, content=agent_response
                )
                execution_time = (
                    datetime.now() - execution_start
                ).total_seconds()
            else:
                logger.info(
                    "Task execution skipped (execute_task=False)"
                )

            total_time = (datetime.now() - start_time).total_seconds()

            result = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "task": {
                    "original": task,
                    "modified": (
                        final_task
                        if boss_response.modified_task
                        else None
                    ),
                },
                "boss_decision": {
                    "selected_agent": boss_response.selected_agent,
                    "reasoning": boss_response.reasoning,
                },
                "execution": {
                    "agent_name": selected_agent.name,
                    "agent_id": selected_agent.id,
                    "was_executed": self.execute_task,
                    "response": (
                        agent_response if self.execute_task else None
                    ),
                    "execution_time": (
                        execution_time if self.execute_task else None
                    ),
                },
                "total_time": total_time,
            }

            logger.info(
                f"Successfully routed task to {selected_agent.name}"
            )
            return result

        except Exception as e:
            logger.error(f"Error routing task: {str(e)}")
            raise

    def run(self, task: str):
        """Route a task to the appropriate agent and return the result"""
        return self.route_task(task)

    def __call__(self, task: str):
        """Route a task to the appropriate agent and return the result"""
        return self.route_task(task)

    def batch_run(self, tasks: List[str] = []):
        """Batch route tasks to the appropriate agents"""
        results = []
        for task in tasks:
            try:
                result = self.route_task(task)
                results.append(result)
            except Exception as e:
                logger.error(f"Error routing task: {str(e)}")
        return results

    def concurrent_batch_run(self, tasks: List[str] = []):
        """Concurrently route tasks to the appropriate agents"""
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor

        results = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.route_task, task)
                for task in tasks
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error routing task: {str(e)}")
        return results
