# import the necessary packages
from fastapi import FastAPI, File, UploadFile, Form, Body
from pydantic import BaseModel, parse_obj_as
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from PIL import Image
from typing import List, Dict
import numpy as np
import os
import app.helpers as helpers
import redis
import uuid
import time
import json
import io
import logging
import cv2

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://fast.localhost",
    "http://fast.localhost:" + os.getenv("SERVER_PORT"),
    "http://localhost",
    "http://localhost:" + os.getenv("SERVER_PORT"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="/static-files"), name="static")

db = redis.StrictRedis(host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"), db=os.getenv("REDIS_DB"))
db.ping() 

print(f"connected to redis: {os.getenv('REDIS_HOST')}") 

class SamPrompt(BaseModel):
    x1 : int
    x2 : int
    w : int = -1
    h : int = -1

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = np.array(image).astype('float32')
    # convert to BGR, because that's what the model expects
    image = image[:, :, ::-1].copy()
    image = np.expand_dims(image, axis=0)
    # return the processed image
    return image

@app.get("/")
async def homepage():
    return FileResponse('/static-files/index.html')

@app.post("/sam/")
def predict_sam(data: str = Form(...), file: UploadFile = File(...)):
    logger.info(data)
    
    #parsed_data = parse_obj_as(List[SamPrompt], data)
    return_data = {"success": False}
    logger.info(f"Prompts: {data}")
    data = json.loads(data)
    contents = file.file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array,cv2.IMREAD_COLOR)

    #image = Image.open(file.file)
    height, width, channels = image.shape
    # if the image mode is not RGB, convert it
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #image = np.array(image).astype('float32')
    #image = np.array(image).astype('uint8')
    # convert to BGR, because that's what the model expects
    #image = image[:, :, ::-1].copy()
    image = np.expand_dims(image, axis=0)
    logger.info(f"Image original dimesions: {width}x{height}")
    # ensure our NumPy array is C-contiguous as well,
    # otherwise we won't be able to serialize it
    image = image.copy(order="C")

    k = str(uuid.uuid4())
    image = helpers.base64_encode_image(image)
    d = {"id": k, "image": image, "height": height, "width": width, "prompts": data}
    db.rpush("image_queue_sam", json.dumps(d))

    while True:
        # attempt to grab the output predictions
        output = db.get(k)

        # check to see if our model has classified the input
        # image
        # print(f"  - output: {output}")
        if output is not None:
            # add the output predictions to our data
            # dictionary so we can return it to the client
            output = output.decode("utf-8")
            return_data["predictions"] = json.loads(output)

            # delete the result from the database and break
            # from the polling loop
            db.delete(k)
            break

        # sleep for a small amount to give the model a chance
        # to classify the input image
        time.sleep(float(os.getenv("CLIENT_SLEEP")))

    # indicate that the request was a success
    return_data["success"] = True
    return_data = jsonable_encoder(return_data)
    # return the data dictionary as a JSON response
    return JSONResponse(content=return_data)

@app.post("/predictor/")
def predict(model_type: str = Form(...), file: UploadFile = File(...)):
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    # read the image in PIL format and prepare it for
    # classification
    #image = flask.request.files["image"].read()
    #image = Image.open(io.BytesIO(file.file))
    logger.info(model_type)
    image = Image.open(file.file)
    width, height = image.size
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype('float32')
    # convert to BGR, because that's what the model expects
    image = image[:, :, ::-1].copy()
    image = np.expand_dims(image, axis=0)
    logger.info(f"Image original dimesions: {width}x{height}")
    # ensure our NumPy array is C-contiguous as well,
    # otherwise we won't be able to serialize it
    image = image.copy(order="C")

    # generate an ID for the classification then add the
    # classification ID + image to the queue
    k = str(uuid.uuid4())
    image = helpers.base64_encode_image(image)
    d = {"id": k, "image": image, "height": height, "width": width}
    db.rpush(model_type, json.dumps(d))
    # keep looping until our model server returns the output
    # predictions
    while True:
        # attempt to grab the output predictions
        output = db.get(k)

        # check to see if our model has classified the input
        # image
        # print(f"  - output: {output}")
        if output is not None:
            # add the output predictions to our data
            # dictionary so we can return it to the client
            output = output.decode("utf-8")
            data["predictions"] = json.loads(output)

            # delete the result from the database and break
            # from the polling loop
            db.delete(k)
            break

        # sleep for a small amount to give the model a chance
        # to classify the input image
        time.sleep(float(os.getenv("CLIENT_SLEEP")))

    # indicate that the request was a success
    data["success"] = True
    return_data = jsonable_encoder(data)
    # return the data dictionary as a JSON response
    return JSONResponse(content=return_data)

# for debugging purposes, it's helpful to start the Flask testing
# server (don't use this for production)
if __name__ == "__main__":
    print("* Starting web service...")
    app.run()
