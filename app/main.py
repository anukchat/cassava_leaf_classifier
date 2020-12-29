from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import ValidationError
import uvicorn

from PIL import Image
import io
import sys
import logging
import cv2
import numpy as np

from response_dto.prediction_response_dto import PredictionResponseDto
from ml.predictions.classify_image import ImageClassifier

app = FastAPI()

origins = [
    "http://127.0.0.1:3000",
    "https://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_classifier = ImageClassifier()


@app.post("/predict/", response_model=PredictionResponseDto)
async def predict(file: UploadFile = File(...)):
    if file.content_type.startswith('image/') is False:
        raise HTTPException(
            status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        contents = await file.read()

        image = np.fromstring(contents, np.int8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        predicted_class = image_classifier.predict(image)

        logging.info(f"Predicted Class: {predicted_class}")
        return {
            "filename": file.filename,
            "contenttype": file.content_type,
            "likely_class": predicted_class
        }

    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))

    except ValidationError as e:
        print(e.json())
