from typing import Dict, List
import openai
from app.core.config import settings
from app.services.file_service import FileService
import json
class NLPService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.file_service = FileService()

    async def parse_query(self, query: str, session_id: str) -> Dict:
        """Parse user query and determine required analysis steps"""
        if session_id:
            metadata = await self.file_service.get_metadata(session_id)
            system_prompt = f"""
            You are a data analysis assistant. Your task is to:
            1. Interpret the user's question about their data
            2. Break it down into specific analysis tasks
            3. Determine appropriate visualizations
            4. Return a structured analysis plan
            
            Metadata of the user's data file: {metadata}

            Return the plan in the following JSON format:
            {{
                "analysis_type": ["statistical", "temporal", "comparative", etc.],
                "required_columns": ["col1", "col2"],
                "operations": ["mean", "correlation", "groupby", etc.],
                "visualizations": ["line_plot", "scatter_plot", etc.],
                "explanation_focus": ["trends", "patterns", "outliers", etc.]
            }}
            """
        else:
            raise ValueError("Please upload your data file first.")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"}
            ],
            temperature=0
        )
        
        # Parse and validate the response
        try:
            analysis_plan = json.loads(response.choices[0].message.content)
            # Ensure session_id is included in the plan
            if isinstance(analysis_plan, dict):
                analysis_plan["session_id"] = session_id
            return analysis_plan
        except Exception as e:
            raise ValueError(f"Failed to parse query: {str(e)}")

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