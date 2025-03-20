import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image

# ✅ 현재 스크립트(=main.py)가 있는 디렉터리 경로 가져오기
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ AI 모델 경로 설정 (현재 폴더에 있는 best.pt)
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# ✅ FastAPI 앱 생성
app = FastAPI()

# ✅ CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ YOLOv5 모델 로드
print("🔄 모델 로딩 중...")
try:
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)
    print("✅ 모델 로딩 완료!")
except Exception as e:
    print(f"🚨 모델 로딩 실패: {e}")
    model = None  # 모델이 로드되지 않으면 None으로 설정

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    if model is None:
        print("🚨 모델이 로드되지 않았습니다!")
        return {"error": "Model not loaded"}

    try:
        print(f"📥 이미지 받음: {file.filename}")

        # 이미지 열기
        image = Image.open(file.file)

        # 모델 실행
        results = model(image)

        # ✅ 감지된 객체 리스트 추출
        detected_classes = results.pandas().xyxy[0]["name"].tolist()
        print("🔍 감지된 객체:", detected_classes)

        # ✅ 결과 설정
        if "basket" in detected_classes:
            response = {"filename": file.filename, "result": "valid"}
        else:
            response = {"filename": file.filename, "result": "invalid"}

    except Exception as e:
        print("🚨 AI 모델 처리 중 오류 발생:", str(e))
        response = {"filename": file.filename, "result": "error", "message": str(e)}

    print("🚀 최종 응답:", response)
    return response
