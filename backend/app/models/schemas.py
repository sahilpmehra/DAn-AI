from pydantic import BaseModel

class AnalysisRequest(BaseModel):
    message: str
    session_id: str | None = None 

class DataSummaryRequest(BaseModel):
    session_id: str | None = None 