from typing import Dict, List
import openai
from app.core.config import settings
from app.services.file_service import FileService
import json
from pydantic import BaseModel, Field

class AnalysisStep(BaseModel):
    id: int
    operation: str
    description: str
    depends_on: List[int] = Field(default_factory=list)
    required_columns: List[str] = Field(default_factory=list)

class AnalysisPlan(BaseModel):
    steps: List[AnalysisStep]

class NLPService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.file_service = FileService()

    async def parse_query(self, query: str, session_id: str) -> Dict:
        """Parse user query into structured analysis steps"""
        system_prompt = """You are a data analysis planner. Break down the user's query into executable steps. 
        Each step should have the following fields:
           - id: Step identifier
           - operation: Name of the operation
           - description: Description of what the step does
           - depends_on: List of step IDs this step depends on
           - required_columns: Columns from the dataset needed for the operation
        
        Focus on creating a logical sequence of operations that build upon each other.
        For complex operations that don't fit into standard categories, mark them as custom operations."""

        user_prompt = f"""Query: {query}

        Let's think step by step. Return a detailed analysis plan as JSON."""

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=AnalysisPlan
        )

        plan = response.choices[0].message

        if plan.refusal:
            return plan.refusal

        return AnalysisPlan.model_validate_json(plan.parsed).model_dump()