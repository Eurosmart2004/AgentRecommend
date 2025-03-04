import json
import os
import litellm
from typing import List
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

class LiteLLMFunctionCaller:
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        base_model: BaseModel = None,
        api_key: str = None,
        temperature: float = 0.1,
        max_tokens: int = 5000,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.api_key = api_key or os.getenv("LITELLM_API_KEY")
        self.temperature = temperature
        self.base_model = base_model
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("API key is required. Set it via env variable or pass it explicitly.")

    def run(self, task: str):
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task},
                ],
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content:str = response["choices"][0]["message"]["content"]
            
            # Remove Markdown formatting if it exists
            if content.startswith("```json"):
                content = content.strip("```json").strip("```").strip()

            parsed_content = json.loads(content)  # Parse into a dictionary
            
            if self.base_model:
                return self.base_model.model_validate(parsed_content)  # Validate using Pydantic model
            
            return parsed_content  # Return as a dictionary if no Pydantic model is provided

        except Exception as e:
            print(f"Error calling {self.model_name}: {e}")
            return None


    def batch_run(self, tasks: List[str]) -> List[BaseModel]:
        return [self.run(task) for task in tasks]

    def concurrent_run(self, tasks: List[str]) -> List[BaseModel]:
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            return list(executor.map(self.run, tasks))