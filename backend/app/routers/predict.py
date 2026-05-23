from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
import shutil, traceback
import os, json
import hashlib

from datetime import datetime
from src.ai_pipeline import processImage

router = APIRouter()

def readFile(file: UploadFile):
    try:
        basePath = "data_source/collected/"
        os.makedirs(basePath, exist_ok=True)
        content = file.file.read()
        imageHash = hashlib.md5(content).hexdigest()
        _, fileExtension = os.path.splitext(file.filename)
        imageId = f"{imageHash}{fileExtension}"
        filePath = os.path.join(basePath, imageId)
        with open(filePath, "wb") as f:
            f.write(content)

        isDuplicate = os.path.exists(os.path.join("data_source/collected/low_confidence/images", imageId))

        return filePath, imageId, isDuplicate
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error reading file")
    finally:
        file.file.close()


def handleLowConfidence(lowConfidenceObjects: list, imageId: str, filePath: str):
    try:
        lowConfDir = "data_source/collected/low_confidence/"
        imgDir = os.path.join(lowConfDir, "images")
        os.makedirs(imgDir, exist_ok=True)
        logFile = os.path.join(lowConfDir, "log_low_confidence.json")

        shutil.copy(filePath, os.path.join(imgDir, imageId))

        log = {}
        if os.path.exists(logFile) and os.path.getsize(logFile) > 0:
            with open(logFile, "r", encoding="utf-8") as f:
                try:
                    log = json.load(f)
                except json.JSONDecodeError:
                    log = {}

        log[imageId] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "issues": lowConfidenceObjects
        }
        with open(logFile, "w", encoding = "utf-8") as f:
            json.dump(log, f, indent=4, ensure_ascii=False)
        print(f"Logged {len(lowConfidenceObjects)} low confidence objects for {imageId}")

    except Exception:
        print("Error handling low confidence objects")
        traceback.print_exc()

@router.post("/api/v1/predict")
async def predict(file: UploadFile = File(...)):
    filePath, imageId, isDuplicate = readFile(file)
    results, lowConfidenceObjects = processImage(filePath)

    if not isDuplicate and len(lowConfidenceObjects) > 0:
        handleLowConfidence(lowConfidenceObjects, imageId, filePath)

    if os.path.exists(filePath):
        os.remove(filePath)

    return {"status": "success", "data": results, "imageId": imageId}
