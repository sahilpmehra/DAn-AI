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
        analysis_plan = await nlp_service.parse_query(request.message)
        
        # Execute the analysis
        result = await analysis_service.execute_analysis(analysis_plan)
        
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 