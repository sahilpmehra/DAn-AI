from typing import Dict, List
import openai
from app.core.config import settings
from app.services.file_service import FileService
import json
from pydantic import BaseModel, Field

class AnalysisStep(BaseModel):
    id: int
    operation: str
    operation_type: str = Field(default="predefined")  # "predefined" or "custom"
    description: str
    depends_on: List[int] = Field(default_factory=list)
    parameters: Dict = Field(default_factory=dict)

class AnalysisPlan(BaseModel):
    steps: List[AnalysisStep]
    required_columns: List[str]
    analysis_type: List[str]

class NLPService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.file_service = FileService()

    async def parse_query(self, query: str, session_id: str) -> Dict:
        """Parse user query into structured analysis steps"""
        system_prompt = """You are a data analysis planner. Break down the user's query into executable steps. 
        Return a JSON plan with:
        1. steps: List of analysis steps with:
           - id: Step identifier
           - operation: Name of the operation
           - operation_type: Either "predefined" for standard operations (statistical, temporal, comparative, clustering) 
             or "custom" for complex operations requiring custom code
           - description: Description of what the step does
           - depends_on: List of step IDs this step depends on
           - parameters: Parameters (columns from the dataset) needed for the operation
        3. analysis_type: List of analysis categories (e.g., statistical, temporal, comparative)
        
        Focus on creating a logical sequence of operations that build upon each other.
        For complex operations that don't fit into standard categories, mark them as custom operations."""

        user_prompt = f"""Query: {query}
        Session ID: {session_id}
        
        Return a detailed analysis plan as JSON."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={ "type": "json_object" }
        )

        plan = response.choices[0].message.content
        return AnalysisPlan.parse_raw(plan).dict()

    def _validate_analysis_plan(self, plan: Dict) -> bool:
        """Validate the analysis plan structure"""
        required_keys = [
            "analysis_type",
            "required_columns",
            "operations",
            "visualizations",
            "explanation_focus"
        ]
        return all(key in plan for key in required_keys) 