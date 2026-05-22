from ultralytics import YOLO
import os, json

currentDir = os.path.dirname(__file__)
vocabPath = os.path.join(currentDir, "../data_source/vocab.json")

# đọc từ điển json
def loadVocabDB():
    with open(vocabPath, "r", encoding="utf-8") as f:
        return json.load(f)

def processImage(image_path: str):
    model = YOLO("../models/yolov8m.pt")
    vocabDb = loadVocabDB()
    
    results = model(image_path, conf=0.5)
    detected_words = set()
    for result in results:
        boxes = result.boxes
        for box in boxes:
            classId = int(box.cls[0])
            detected_words.add(model.names[classId])
            
    final_output = []
    for word in detected_words:
        info = vocabDb.get(word, {"related": ["object"], "example": f"I see a {word}."})
        final_output.append({
            "word": word,
            "related": info["related"],
            "example": info["example"]
        })
        
    return final_output