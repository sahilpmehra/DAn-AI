from fastapi import APIRouter, HTTPException
from app.services.data_summary_service import DataSummaryService
from app.models.schemas import DataSummaryRequest
from typing import Dict
import json
import traceback
import numpy as np

router = APIRouter()
data_summary_service = DataSummaryService()

@router.post("/")
async def summarize_data(request: DataSummaryRequest) -> Dict:
    try:
        # Get summary data from service
        result = await data_summary_service.get_summary_data(request.session_id)
        
        # Use custom JSON encoder for numpy types
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
        
        return json.loads(json.dumps(result, cls=NumpyEncoder))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) 