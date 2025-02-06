from typing import List, Optional
from pydantic import BaseModel, Field

class AnalysisRequest(BaseModel):
    message: str
    session_id: str | None = None 

class DataSummaryRequest(BaseModel):
    session_id: str | None = None 

class BarChartSchema(BaseModel):
    labels: List[str] = Field(default_factory=list)  # x-axis labels
    datasets: List[dict] = Field(default_factory=list)  # Contains data and styling
    options: Optional[dict] = None  # Optional chart configuration

class LineChartSchema(BaseModel):
    labels: List[str] = Field(default_factory=list)  # x-axis labels
    datasets: List[dict] = Field(default_factory=list)  # Contains data points and styling
    options: Optional[dict] = None

class PieChartSchema(BaseModel):
    labels: List[str] = Field(default_factory=list)  # segment labels
    datasets: List[dict] = Field(default_factory=list)  # Contains values and colors
    options: Optional[dict] = None

class TableSchema(BaseModel):
    headers: List[str] = Field(default_factory=list)  # column headers
    rows: List[List[str | int | float]] = Field(default_factory=list)  # table data
    options: Optional[dict] = None

class OtherChartSchema(BaseModel):
    data: dict = Field(default_factory=dict)  # Generic data structure
    options: Optional[dict] = None