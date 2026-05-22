from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(title = "English Vocab AI - MLOps Project")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)

# lấy path đến thư mục frontend
frontend_path = os.path.join(os.path.dirname(__file__), "../../frontend")

# gọi các file tĩnh trong frontend (yêu cầu lấy file)
# hiểu đơn giản là cấp quyền truy cập cho fastapi để vào thư mục frontend lấy file 
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))
