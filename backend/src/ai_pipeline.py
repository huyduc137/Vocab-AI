from ultralytics import YOLO
import os, json, cv2, base64

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

    img = cv2.imread(image_path)

    detectedWords = {}
    lowConfidenceObjects = []        # Danh sách để lưu các đối tượng có độ tin cậy thấp
    for result in results:
        for box in result.boxes:
            classId = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[classId]
            if conf >= 0.7:
                if label not in detectedWords:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cropped_img = img[y1:y2, x1:x2]
                    _, buffer = cv2.imencode('.jpg', cropped_img)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    detectedWords[label] = img_base64
            else:
                lowConfidenceObjects.append(
                    {
                        "word": label,
                        "confidence": round(conf, 2)
                    }
                )
                
            
    final_output = []
    for word, img_b64 in detectedWords.items():
        info = vocabDb.get(word, {"related": ["object"], "example": f"I see a {word}."})
        final_output.append({
            "word": word,
            "related": info["related"],
            "example": info["example"],
            "cropped_image": img_b64
        })
        
    return final_output, lowConfidenceObjects