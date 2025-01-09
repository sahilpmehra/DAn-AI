from fastapi import APIRouter, HTTPException
from app.services.nlp_service import NLPService
from app.services.analysis_service import AnalysisService
from app.models.schemas import AnalysisRequest
from typing import Dict

router = APIRouter()
nlp_service = NLPService()
analysis_service = AnalysisService()

@router.post("/")
async def analyze_data(request: AnalysisRequest) -> Dict:
    try:
        # Parse the user query
        # analysis_plan = await nlp_service.parse_query(request.message, request.session_id)
        analysis_plan = {'analysis_type': ['statistical', 'descriptive'], 'required_columns': ['acceptance', 'difficulty', 'frequency'], 'operations': ['mean', 'count', 'unique', 'describe'], 'visualizations': ['bar_chart', 'box_plot'], 'explanation_focus': ['summary statistics', 'distribution of values', 'categorical breakdown'], 'session_id': 'f06875b2-7555-4fb9-8994-34c8e964cb06'}

        # Execute the analysis
        result = await analysis_service.execute_analysis(analysis_plan, request.session_id)
        
        return {"response": result}
    except Exception as e:
        print("This is where the error is happening: ", e)
        raise HTTPException(status_code=500, detail=str(e)) 