from typing import List, Optional
from pydantic import BaseModel

class AnalysisRequest(BaseModel):
    message: str
    session_id: str | None = None 

class DataSummaryRequest(BaseModel):
    session_id: str | None = None 

class BarChartSchema(BaseModel):
    labels: List[str]  # x-axis labels
    datasets: List[dict]  # Contains data and styling
    options: Optional[dict] = None  # Optional chart configuration

class LineChartSchema(BaseModel):
    labels: List[str]  # x-axis labels
    datasets: List[dict]  # Contains data points and styling
    options: Optional[dict] = None

class PieChartSchema(BaseModel):
    labels: List[str]  # segment labels
    datasets: List[dict]  # Contains values and colors
    options: Optional[dict] = None

class TableSchema(BaseModel):
    headers: List[str]  # column headers
    rows: List[List[str | int | float]]  # table data
    options: Optional[dict] = None

class OtherChartSchema(BaseModel):
    data: dict  # Generic data structure
    options: Optional[dict] = None