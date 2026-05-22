from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
import shutil, traceback
import os
from src.ai_pipeline import processImage

router = APIRouter()
@router.post("/api/v1/predict")
async def predict(file: UploadFile = File(...)):
    try:
        filePath = f"data_source/collected/{file.filename}"
        os.makedirs("data_source/collected", exist_ok=True)
        with open(filePath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = processImage(filePath)
        return {"status": "success", "data": results}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))