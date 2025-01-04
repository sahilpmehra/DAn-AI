from typing import Dict, List
import anthropic
from app.core.config import settings

class NLPService:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.ANTHROPIC_MODEL
        
    async def parse_query(self, query: str) -> Dict:
        """Parse user query and determine required analysis steps"""
        system_prompt = """
        You are a data analysis assistant. Your task is to:
        1. Interpret the user's question about their data
        2. Break it down into specific analysis tasks
        3. Determine appropriate visualizations
        4. Return a structured analysis plan
        
        Return the plan in the following JSON format:
        {
            "analysis_type": ["statistical", "temporal", "comparative", etc.],
            "required_columns": ["col1", "col2"],
            "operations": ["mean", "correlation", "groupby", etc.],
            "visualizations": ["line_plot", "scatter_plot", etc.],
            "explanation_focus": ["trends", "patterns", "outliers", etc.]
        }
        """
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=messages,
            temperature=0
        )
        
        # Parse and validate the response
        try:
            analysis_plan = response.content[0].text
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