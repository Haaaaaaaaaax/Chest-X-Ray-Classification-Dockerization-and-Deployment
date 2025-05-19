from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from uuid import uuid4
import traceback

# Importing model-specific predict functions
from DINOv2_Inference import predict as dino_predict
from efficentnetb3_Keras_Inference import predict as effnet_predict
from CNN_from_skratch_Inference import predict as cnn_predict

app = FastAPI()

# Directory to store uploaded files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Utility function to save uploaded file
def save_upload_file(upload_file: UploadFile) -> str:
    file_ext = os.path.splitext(upload_file.filename)[-1]
    file_path = os.path.join(UPLOAD_DIR, f"{uuid4()}{file_ext}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

# DINOv2 model endpoint
@app.post("/model1")
async def infer_model1(file: UploadFile = File(...)):
    try:
        image_path = save_upload_file(file)
        predicted_label, confidence = dino_predict(image_path)
        return JSONResponse(content={
            "model": "DINOv2",
            "label": predicted_label,
            "confidence": float(confidence)
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# EfficientNetB3 model endpoint
@app.post("/model2")
async def infer_model2(file: UploadFile = File(...)):
    try:
        image_path = save_upload_file(file)
        predicted_label, confidence = effnet_predict(image_path)
        return JSONResponse(content={
            "model": "EfficientNetB0",
            "label": predicted_label,
            "confidence": float(confidence)
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# CNN from scratch model endpoint
@app.post("/model3")
async def infer_model3(file: UploadFile = File(...)):
    try:
        image_path = save_upload_file(file)
        predicted_label, confidence = cnn_predict(image_path)
        return JSONResponse(content={
            "model": "CNN_from_scratch",
            "label": predicted_label,
            "confidence": float(confidence)
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
