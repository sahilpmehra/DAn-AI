from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.services.nlp_service import NLPService
from app.services.analysis_service import AnalysisService
from app.models.schemas import AnalysisRequest
from typing import Dict
import json
import traceback
import numpy as np

router = APIRouter()
nlp_service = NLPService()
analysis_service = AnalysisService()

@router.post("/")
async def analyze_data(request: AnalysisRequest, background_tasks: BackgroundTasks) -> Dict:
    try:
        # Parse the user query and generate analysis plan
        # analysis_plan = await nlp_service.parse_query(request.message, request.session_id)
        analysis_plan = {'steps': [{'id': '1', 'operation': 'Group by Difficulty', 'description': "Group the dataset by the 'difficulty' column to count the number of questions in each category (easy, medium, hard).", 'depends_on': [], 'required_columns': ['difficulty'], 'has_visualization': False, 'visualization_type': None}, {'id': '2', 'operation': 'Count Questions', 'description': 'Count the number of occurrences of each difficulty level from the grouped data.', 'depends_on': [1], 'required_columns': ['difficulty'], 'has_visualization': False, 'visualization_type': None}, {'id': '3', 'operation': 'Prepare Data for Visualization', 'description': 'Organize the counted data into a format suitable for creating a bar chart, typically a DataFrame with difficulty levels as one column and counts as another.', 'depends_on': [2], 'required_columns': [], 'has_visualization': False, 'visualization_type': None}, {'id': '4', 'operation': 'Create Bar Chart', 'description': 'Generate a bar chart visualization of the distribution of questions across different difficulty levels using the prepared data.', 'depends_on': [3], 'required_columns': [], 'has_visualization': True, 'visualization_type': 'Bar'}]}

        # If analysis_plan is a string, it means it's a refusal message
        if isinstance(analysis_plan, str):
            return {"response": json.dumps({
                "refusal": analysis_plan,
                "results": None
            })}

        # Execute the analysis
        # result = await analysis_service.execute_analysis(analysis_plan, request.session_id)
        result = {'1': {'step_result': {'data': [{'difficulty': 'Medium', 'count': 13}, {'difficulty': 'Hard', 'count': 8}], 'chart_data': None, 'stats': {'total_questions': 21, 'difficulty_categories': 2}, 'error': None}, 'metadata': {'description': "Group the dataset by the 'difficulty' column to count the number of questions in each category (easy, medium, hard).", 'columns_used': ['difficulty']}}, '2': {'step_result': {'data': [{'difficulty': 'Medium', 'count': 1}, {'difficulty': 'Hard', 'count': 1}], 'chart_data': None, 'stats': {'total_difficulties': 2, 'source_data_stats': {'total_questions': 21, 'difficulty_categories': 2}}, 'error': None}, 'metadata': {'description': 'Count the number of occurrences of each difficulty level from the grouped data.', 'columns_used': ['difficulty']}}, '3': {'step_result': {'data': [{'difficulty': 'Medium', 'count': 1}, {'difficulty': 'Hard', 'count': 1}], 'chart_data': None, 'stats': {'total_difficulties': 2, 'source_data_stats': {'total_difficulties': 2, 'source_data_stats': {'total_questions': 21, 'difficulty_categories': 2}}}, 'error': None}, 'metadata': {'description': 'Organize the counted data into a format suitable for creating a bar chart, typically a DataFrame with difficulty levels as one column and counts as another.', 'columns_used': []}}, '4': {'step_result': {'data': [{'difficulty': 'Medium', 'count': 1}, {'difficulty': 'Hard', 'count': 1}], 'chart_data': {'labels': ['Medium', 'Hard'], 'datasets': [{'label': 'Question Distribution by Difficulty', 'data': [1, 1], 'backgroundColor': ['#FF6384', '#36A2EB', '#FFCE56']}], 'options': None}, 'stats': {'total_difficulties': 2}, 'error': None}, 'metadata': {'description': 'Generate a bar chart visualization of the distribution of questions across different difficulty levels using the prepared data.', 'columns_used': []}}}
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                    np.int16, np.int32, np.int64, np.uint8,
                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        
        result_json = json.dumps({
            'analysis_plan': analysis_plan,
            'results': result
        }, cls=NumpyEncoder)
        
        return {"response": result_json}
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) 