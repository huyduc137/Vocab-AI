# English Vocab AI

English Vocab AI là ứng dụng web hỗ trợ học từ vựng tiếng Anh thông qua hình ảnh. Người dùng tải lên một bức ảnh bất kỳ, hệ thống dùng mô hình YOLOv8 để nhận diện các vật thể trong ảnh, cắt riêng từng vật thể và hiển thị từ vựng tiếng Anh tương ứng kèm từ liên quan, ví dụ mẫu và ảnh minh họa đã được cắt ra.

Dự án không chỉ dừng ở bài toán nhận diện vật thể. Backend được thiết kế theo hướng MLOps đơn giản với cơ chế thu thập các trường hợp mô hình dự đoán có độ tin cậy thấp, phục vụ việc kiểm tra, gán nhãn lại và cải thiện mô hình trong tương lai.

## Tính năng chính

- Tải ảnh từ trình duyệt và gửi về backend để phân tích.
- Nhận diện vật thể bằng mô hình YOLOv8.
- Cắt vùng ảnh của từng vật thể bằng OpenCV.
- Trả ảnh vật thể đã cắt dưới dạng Base64 để frontend hiển thị trực tiếp.
- Tra cứu từ vựng, từ liên quan và câu ví dụ từ file `vocab_db.json`.
- Ghi nhận các vật thể có độ tin cậy thấp vào thư mục thu thập dữ liệu.
- Giao diện frontend thuần HTML, CSS và JavaScript, không phụ thuộc framework nặng.

## Cấu trúc thư mục

```text
Project_Vocab_AI/
|-- backend/
|   |-- app/
|   |   |-- main.py
|   |   `-- routers/
|   |       `-- predict.py
|   |-- src/
|   |   |-- ai_pipeline.py
|   |   |-- data_processing.py
|   |   `-- model_training.py
|   |-- data_source/
|   |   |-- vocab_db.json
|   |   `-- collected/
|   |       `-- low_confidence/
|   |           |-- images/
|   |           `-- log_low_confidence.json
|   |-- models/
|   |   `-- yolov8m.pt
|   `-- requirements.txt
|-- frontend/
|   |-- index.html
|   |-- script.js
|   `-- style.css
|-- docs/
|-- notebooks/
|-- docker/
`-- README.md
```


## Cài đặt

Tạo và kích hoạt môi trường ảo:

```bash
python -m venv venv
```

Trên Windows:

```bash
venv\Scripts\activate
```

Trên macOS/Linux:

```bash
source venv/bin/activate
```

Cài đặt thư viện backend:

```bash
cd backend
pip install -r requirements.txt
```

Nếu không dùng file `requirements.txt`, có thể cài các thư viện chính:

```bash
pip install fastapi "uvicorn[standard]" python-multipart "numpy<2" opencv-python==4.9.0.80 ultralytics
```

## Khởi chạy ứng dụng

Chạy server FastAPI từ thư mục `backend`:

```bash
uvicorn app.main:app --reload
```

Sau khi server khởi động, mở trình duyệt tại:

```text
http://127.0.0.1:8000
```

Frontend được FastAPI phục vụ trực tiếp tại route `/`, còn các file tĩnh được mount tại `/static`.

## API chính

### Phân tích ảnh

```http
POST /api/v1/predict
```

Request dùng `multipart/form-data` với field:

| Field | Kiểu | Mô tả |
| --- | --- | --- |
| `file` | image file | Ảnh người dùng tải lên |

Response mẫu:

```json
{
  "status": "success",
  "imageId": "example_hash.jpg",
  "data": [
    {
      "word": "car",
      "related": ["vehicle", "drive", "road"],
      "example": "The car is parked on the street.",
      "cropped_image": "base64_image_string"
    }
  ]
}
```

## Tech stack

### Backend

- **FastAPI**: xây dựng API, phục vụ frontend tĩnh và xử lý upload ảnh.
- **Uvicorn**: ASGI server dùng để chạy ứng dụng FastAPI.
- **Ultralytics YOLOv8**: mô hình deep learning dùng để nhận diện vật thể.
- **OpenCV**: đọc ảnh, xử lý ma trận ảnh, cắt vùng vật thể và encode ảnh.
- **Python Multipart**: hỗ trợ nhận file upload qua form data.

### Frontend

- **HTML5**: cấu trúc giao diện.
- **CSS3**: định dạng bố cục và giao diện người dùng.
- **JavaScript Fetch API**: gửi ảnh lên backend và render kết quả trả về.

### Dữ liệu

- **`vocab_db.json`**: cơ sở dữ liệu từ vựng cục bộ, ánh xạ nhãn vật thể sang từ liên quan và câu ví dụ.
- **`log_low_confidence.json`**: log các trường hợp mô hình nhận diện với độ tin cậy thấp.

## Kỹ thuật lõi

### 1. Deep Learning Inference

Ảnh đầu vào được truyền qua pipeline trong `backend/src/ai_pipeline.py`. Mô hình YOLOv8 trả về danh sách vật thể, nhãn dự đoán, bounding box và confidence score. Các vật thể có confidence từ `0.7` trở lên được đưa vào danh sách kết quả chính.

### 2. Cắt vùng ảnh bằng Matrix Slicing

Từ bounding box `[x_min, y_min, x_max, y_max]`, hệ thống cắt trực tiếp trên ma trận ảnh OpenCV:

```python
cropped_img = img[y1:y2, x1:x2]
```

Cách làm này nhanh, chính xác ở mức pixel và không cần dùng công cụ xử lý ảnh trung gian.

### 3. Base64 In-Memory Encoding

Ảnh vật thể sau khi cắt được encode trực tiếp sang Base64 trên RAM:

```python
_, buffer = cv2.imencode(".jpg", cropped_img)
img_base64 = base64.b64encode(buffer).decode("utf-8")
```

Chuỗi Base64 được trả về trong JSON để frontend hiển thị bằng thẻ `<img>`, tránh phải lưu nhiều file ảnh tạm trên server.

### 4. MD5 Hashing cho ảnh upload

Backend tính MD5 từ nội dung file ảnh để tạo `imageId`. Cách này giúp định danh ảnh theo nội dung, hỗ trợ kiểm tra trùng lặp khi lưu các ảnh có độ tin cậy thấp.

### 5. Data Flywheel cho MLOps

Các vật thể có confidence thấp hơn `0.7` được ghi nhận vào:

```text
backend/data_source/collected/low_confidence/
```

Thông tin log được lưu trong `log_low_confidence.json`, còn ảnh gốc được lưu trong thư mục `images/`. Đây là nguồn dữ liệu để kiểm tra lỗi, gán nhãn lại và tái huấn luyện mô hình.

## Luồng hoạt động

1. Người dùng chọn ảnh trên giao diện web.
2. Frontend gửi ảnh đến `POST /api/v1/predict`.
3. Backend lưu tạm ảnh upload và tạo `imageId` bằng MD5.
4. `processImage()` chạy YOLOv8 để nhận diện vật thể.
5. Với vật thể đủ confidence, backend cắt ảnh, encode Base64 và ghép dữ liệu từ vựng.
6. Với vật thể confidence thấp, backend ghi log để phục vụ cải thiện dữ liệu.
7. Frontend render danh sách từ vựng, ảnh cắt, từ liên quan và câu ví dụ.

## Roadmap

- **Text-to-Speech**: thêm phát âm từ vựng bằng Web Speech API.
- **Instance Segmentation**: chuyển từ YOLOv8 bounding box sang YOLOv8 segmentation để tách vật thể chi tiết hơn.
- **Reverse Image Search**: dùng vector embedding và FAISS để tìm ảnh vật thể tương tự.
- **Chatbot học từ vựng**: tích hợp LLM để giải thích từ, tạo hội thoại hoặc câu chuyện ngắn từ các từ đã nhận diện.
- **Dashboard dữ liệu lỗi**: xây giao diện quản lý ảnh low-confidence để hỗ trợ gán nhãn và huấn luyện lại.

