from fastapi import UploadFile
import pandas as pd
import uuid
from typing import Dict
import redis
import json
from io import StringIO
from app.core.config import settings

class FileService:
    def __init__(self):
        # Initialize Redis connection using settings
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD
        )
        self.EXPIRE_TIME = settings.CACHE_EXPIRY

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
        """Store DataFrame in Redis and return session ID"""
        session_id = str(uuid.uuid4())
        
        # Convert DataFrame to JSON for storage
        df_json = df.to_json(orient='split')
        
        # Store in Redis with expiry
        self.redis_client.setex(
            f"dataset:{session_id}",
            self.EXPIRE_TIME,
            df_json
        )
        
        # Store column metadata
        metadata = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'shape': df.shape
        }
        self.redis_client.setex(
            f"metadata:{session_id}",
            self.EXPIRE_TIME,
            json.dumps(metadata)
        )
        
        return session_id

    async def get_dataframe(self, session_id: str) -> pd.DataFrame:
        """Retrieve DataFrame from Redis"""
        df_json = self.redis_client.get(f"dataset:{session_id}")
        if df_json is None:
            raise ValueError("Session expired or not found")
            
        return pd.read_json(df_json, orient='split')

    async def get_metadata(self, session_id: str) -> Dict:
        """Retrieve dataset metadata"""
        metadata_json = self.redis_client.get(f"metadata:{session_id}")
        if metadata_json is None:
            raise ValueError("Session metadata not found")
            
        return json.loads(metadata_json) 