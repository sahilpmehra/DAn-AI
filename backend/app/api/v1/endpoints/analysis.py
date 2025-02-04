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
        analysis_plan = await nlp_service.parse_query(request.message, request.session_id)
        # analysis_plan = {'analysis_type': ['statistical', 'descriptive'], 'required_columns': ['acceptance', 'difficulty', 'frequency'], 'operations': ['mean', 'count', 'unique', 'describe'], 'visualizations': ['bar_chart', 'box_plot'], 'explanation_focus': ['summary statistics', 'distribution of values', 'categorical breakdown'], 'session_id': 'f06875b2-7555-4fb9-8994-34c8e964cb06'}
        
        # If analysis_plan is a string, it means it's a refusal message
        if isinstance(analysis_plan, str):
            return {"response": json.dumps({
                "refusal": analysis_plan,
                "results": None
            })}

        # Execute the analysis
        result = await analysis_service.execute_analysis(analysis_plan, request.session_id)
        
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