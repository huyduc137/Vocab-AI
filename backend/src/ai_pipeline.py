from ultralytics import YOLO
import os, json

currentDir = os.path.dirname(__file__)
vocabPath = os.path.join(currentDir, "../data_source/vocab_db.json")
modelPath = os.path.join(currentDir, "../models/yolov8m.pt")
model = YOLO(modelPath)

# đọc từ điển json
def loadVocabDB():
    with open(vocabPath, "r", encoding="utf-8") as f:
        return json.load(f)

def processImage(image_path: str):
    vocabDb = loadVocabDB()
    
    results = model(image_path, conf=0.1)
    detectedWords = set()
    lowConfidenceObjects = []        # Danh sách để lưu các đối tượng có độ tin cậy thấp
    for result in results:
        for box in result.boxes:
            classId = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[classId]
            if conf >= 0.7:
                detectedWords.add(label)
            else:
                lowConfidenceObjects.append(
                    {
                        "word": label,
                        "confidence": round(conf, 2)
                    }
                )
                
            
    final_output = []
    for word in detectedWords:
        info = vocabDb.get(word, {"related": ["object"], "example": f"I see a {word}."})
        final_output.append({
            "word": word,
            "related": info["related"],
            "example": info["example"]
        })
        
    return final_output, lowConfidenceObjects