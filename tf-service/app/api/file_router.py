import os
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import logging
import cv2

router = APIRouter()
logger = logging.getLogger("tf_main")

# 업로드 디렉토리와 출력 디렉토리를 app 내부로 고정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"파일 업로드 디렉토리: {UPLOAD_DIR}")
logger.info(f"파일 출력 디렉토리: {OUTPUT_DIR}")

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    logger.info(f"파일 업로드 시작: {file.filename}, 저장 위치: {file_location}")
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"파일 업로드 성공: {file.filename}")
        if os.path.exists(file_location):
            file_size = os.path.getsize(file_location)
            logger.info(f"파일 저장 확인: {file_location}, 크기: {file_size} bytes")
        else:
            logger.error(f"파일이 저장되지 않음: {file_location}")
        return JSONResponse(content={"filename": file.filename, "message": "파일 업로드 성공!", "path": file_location})
    except Exception as e:
        logger.error(f"파일 업로드 실패: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 예시: JSON만 받는 predict 엔드포인트
class PredictRequest(BaseModel):
    data: list

@router.post("/predict")
async def predict(request_body: PredictRequest):
    logger.info(f"predict 요청: {request_body}")
    # 실제 예측 로직은 여기에 작성
    return {"result": "ok", "received": request_body.data}

@router.get("/mosaic")
async def file_mosaic():
    girl = os.path.join(BASE_DIR, 'data', 'girl.jpg')
    cascade = os.path.join(BASE_DIR, 'data', 'haarcascade_frontalface_alt.xml')
    face_cascade = cv2.CascadeClassifier(cascade)
    img = cv2.imread(girl)
    face = face_cascade.detectMultiScale(img, minSize=(150,150))
    if len(face) == 0:
        logger.error('얼굴인식 실패')
        return JSONResponse(content={"error": "얼굴인식 실패"}, status_code=400)
    for(x,y,w,h) in face:
        logger.info(f'얼굴의 좌표 = {x}, {y}, {w}, {h}')
        red = (0,0,255)
        cv2.rectangle(img, (x, y), (x+w, y+h), red, thickness=20)
    output_path = os.path.join(OUTPUT_DIR, 'girl-face.png')
    cv2.imwrite(output_path, img)
    return JSONResponse(content={
        "message": "모자이크 처리 완료",
        "output_path": output_path
    })