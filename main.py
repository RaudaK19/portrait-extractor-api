from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import os
import base64


app = FastAPI()


@app.get("/")
def home():
    return {"message": "Portrait Extraction API is running"}


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image)
    gray = cv2.cvtColor (image_array, cv2.COLOR_RGB2GRAY)
    cascade_path = os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml"
    )
    print("Cascade path:", cascade_path)
    print("File exists:", os.path.exists(cascade_path))
    face_cascade = cv2.CascadeClassifier(cascade_path)
    print("Cascade empty:", face_cascade.empty())
    if face_cascade.empty():
        raise RuntimeError("Face detector failed to load")

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )
    if len(faces) ==0: 
        return{
            "faces_detected": 0, 
            "message": "No face deteced"
        }
    x,y,w,h = faces[0]
    cropped_face = image_array[y:y+h, x:x+w]
    cropped_image = Image.fromarray(cropped_face)
    buffer = BytesIO()
    cropped_image.save(
        buffer, 
        format="JPEG"
    )
    base64_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    
    return {
        "faces_detected":len(faces),
        "face_coordinates": faces.tolist(),
        "portrait_base64":base64_image
    }
