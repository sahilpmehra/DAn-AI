from fastapi import UploadFile
import pandas as pd
import uuid
from typing import Dict
import redis
import json
from io import StringIO
from app.core.config import settings

class FileService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize storage only once
            cls._instance.storage = {}
            cls._instance.EXPIRE_TIME = 3600  # 1 hour
        return cls._instance

    async def process_file(self, file: UploadFile) -> pd.DataFrame:
        """Process uploaded file and convert to pandas DataFrame"""
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            # Handle CSV files
            df = pd.read_csv(StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.xlsx'):
            # Handle Excel files
            df = pd.read_excel(content)
        else:
            raise ValueError("Unsupported file format")

        # Basic data cleaning
        df = self._clean_dataframe(df)
        return df

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data cleaning operations"""
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Handle missing values using newer methods
        df = df.ffill().bfill()
        
        # Convert column names to snake_case
        df.columns = [self._to_snake_case(col) for col in df.columns]
        
        return df

    def _to_snake_case(self, name: str) -> str:
        """Convert string to snake_case"""
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    async def store_dataframe(self, df: pd.DataFrame) -> str:
        """Store DataFrame in memory and return session ID"""
        session_id = str(uuid.uuid4())
        
        # Store DataFrame and metadata in memory
        self.storage[f"dataset:{session_id}"] = df.to_json(orient='split')
        self.storage[f"metadata:{session_id}"] = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'shape': df.shape
        }
        
        return session_id

    async def get_dataframe(self, session_id: str) -> pd.DataFrame:
        """Retrieve DataFrame from in-memory storage"""
        df_json = self.storage.get(f"dataset:{session_id}")
        if df_json is None:
            raise ValueError("Session expired or not found")
            
        return pd.read_json(StringIO(df_json), orient='split')

    async def get_metadata(self, session_id: str) -> Dict:
        """Retrieve dataset metadata from in-memory storage"""
        metadata = self.storage.get(f"metadata:{session_id}")
        if metadata is None:
            raise ValueError("Session metadata not found")
            
        return metadata 