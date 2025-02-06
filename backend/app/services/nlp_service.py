from typing import Dict, List, Literal
import openai
from app.core.config import settings
from app.services.file_service import FileService
import json
from pydantic import BaseModel, Field

# Define the VizType
VizType = Literal["Bar", "Line", "Pie", "Table", "Other", None]

class AnalysisStep(BaseModel):
    id: str
    operation: str
    description: str
    depends_on: List[int] = Field(default_factory=list)
    required_columns: List[str] = Field(default_factory=list)
    has_visualization: bool = False
    visualization_type: VizType = None
class AnalysisPlan(BaseModel):
    steps: List[AnalysisStep]

class NLPService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.file_service = FileService()

    async def get_dataset_stats(self, session_id: str) -> Dict:
        """Get basic statistics about the dataset columns"""
        df = await self.file_service.get_dataframe(session_id)
        
        stats = {
            column: {
                "python_type": str(df[column].dtype),
                "data_type": "numeric" if df[column].dtype.kind in 'biufc' 
                            else "categorical" if df[column].dtype.kind in 'O' 
                            else "datetime" if df[column].dtype.kind in 'M' 
                            else "other",
                # "sample_values": df[column].head(3).tolist()
            }
            for column in df.columns
        }
        return stats

    async def parse_query(self, query: str, session_id: str) -> Dict:
        """Parse user query into structured analysis steps"""
        dataset_stats = await self.get_dataset_stats(session_id)
        
        system_prompt = f"""You are a data analysis planner. You will help break down the user's query into executable steps.

        Each step should have the following fields:
           - id: Step identifier e.g. "1", "2", "3" etc.
           - operation: Name of the operation
           - description: Description of what the step does
           - depends_on: List of step IDs this step depends on
           - required_columns: Columns from the dataset needed for the operation
           - has_visualization: Whether the step has a visualization
           - visualization_type: The type of visualization to use in the step i.e. Bar, Line, Pie, Table or Other. It should be None if the step does not have a visualization.
        
        Focus on creating a logical sequence of operations that build upon each other.
        For complex operations that don't fit into standard categories, mark them as custom operations.
        Only suggest operations using columns that exist in the dataset.
        
        The dataset has the following columns and their properties:
        {json.dumps(dataset_stats, indent=2)}"""

        user_prompt = f"""Query: {query}

        Let's think step by step. Return a detailed analysis plan as JSON."""

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=AnalysisPlan
            )

            # Get the parsed response
            plan = response.choices[0].message

            if plan.refusal:
                return plan.refusal

            # Validate the response against our model
            try:
                validated_plan = AnalysisPlan.model_validate(plan.parsed)
                
                # Additional validation: check if required columns exist in dataset
                for step in validated_plan.steps:
                    invalid_columns = [col for col in step.required_columns if col not in dataset_stats]
                    if invalid_columns:
                        raise ValueError(f"Step {step.id} references non-existent columns: {invalid_columns}")
                
                return validated_plan.model_dump()
            
            except ValueError as e:
                raise ValueError(f"Invalid analysis plan structure: {str(e)}")
                
        except Exception as e:
            raise ValueError(f"Failed to parse query: {str(e)}")