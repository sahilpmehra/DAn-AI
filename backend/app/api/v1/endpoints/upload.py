from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.file_service import FileService
from typing import Optional
import pandas as pd

router = APIRouter()
file_service = FileService()

@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    print(file.filename)
    # Validate file extension
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    try:
        # Process the file
        df = await file_service.process_file(file)
        
        # Store the processed data
        session_id = await file_service.store_dataframe(df)
        
        return {"session_id": session_id, "message": "File processed successfully"}
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))