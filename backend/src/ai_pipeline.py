from ultralytics import YOLO

VOCAB_DB = {
    "person": {"related": ["human", "man"], "example": "A person is walking."},
    "dog": {"related": ["pet", "animal"], "example": "The dog is playing."}
}

def processImage(image_path: str):
    model = YOLO("../models/yolov8m.pt")
    
    results = model(image_path, conf=0.5)
    detected_words = set()
    for result in results:
        boxes = result.boxes
        for box in boxes:
            classId = int(box.cls[0])
            detected_words.add(model.names[classId])
            
    final_output = []
    for word in detected_words:
        info = VOCAB_DB.get(word, {"related": ["object"], "example": f"I see a {word}."})
        final_output.append({
            "word": word,
            "related": info["related"],
            "example": info["example"]
        })
        
    return final_output