# import the necessary packages
from fastapi import FastAPI, File, UploadFile, Form, Depends, Response
from pydantic import BaseModel, parse_obj_as
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from .routers import sam_router
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
#from .database import get_db

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

host = os.getenv("REDIS_HOST", "localhost")
port = int(os.getenv("REDIS_PORT", 6379))
db = int(os.getenv("REDIS_DB", 0))

pool = redis.ConnectionPool(host=host,
                            port=port,
                            db=db)

db = redis.Redis(connection_pool=pool)

def get_db():
    db = redis.Redis(connection_pool=pool)
    return db

# Try to connect to Redis
for _ in range(5):
    try:
        db.ping()
        break
    except redis.ConnectionError:
        logger.info("Could not connect to Redis. Retrying in 5 seconds...")
        time.sleep(5)
else:
    raise Exception("Could not connect to Redis.")

app = FastAPI()

app.include_router(sam_router.router, dependencies=[Depends(get_db)])

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
    return_data = {"success": False}
    logger.info(f"Prompts: {data}")
    data = json.loads(data)
    contents = file.file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array,cv2.IMREAD_COLOR)
    height, width, channels = image.shape
    # if the image mode is not RGB, convert it
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
    
    output = json.loads(output)
    indexed_areas = [(i, box['w'] * box['h']) for i, (box, score) in enumerate(zip(output[0].get('box'), output[0].get('score'))) if score > 0.9]
    if not indexed_areas:
        print("no high scores")
        lst = output[0].get('score')
        largest_index = max(range(len(lst)), key=lst.__getitem__)
    else:
        # Get the index and area of the box with the higest score
        indexed_scores = [(i,score) for i, (box,score) in enumerate(zip(output[0].get('box'),output[0].get('score')))]
        largest_index, _ = max(indexed_scores, key=lambda x: x[1])

    bounding_poly = output[0].get('poly')[largest_index]
    bounding_box = output[0].get('box')[largest_index]

    return_data["predictions"][0]['poly'] = bounding_poly
    return_data["predictions"][0]['box'] = bounding_box

    return_data["success"] = True
    return_data = jsonable_encoder(return_data)
    # return the data dictionary as a JSON response
    return JSONResponse(content=return_data)

@app.post("/sam_multi/")
def predict_sam_multi(data: str = Form(...), file: UploadFile = File(...)):
    logger.info(data)
    return_data = {"success": False}
    logger.info(f"Prompts: {data}")
    data = json.loads(data)
    contents = file.file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array,cv2.IMREAD_COLOR)
    height, width, channels = image.shape
    # if the image mode is not RGB, convert it
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

@app.post("/filet_predictor/")
def filet_predict(model_type: str = Form(...), file: UploadFile = File(...)):
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    # read the image in PIL format and prepare it for
    # classification
    logger.info(model_type)
    image = Image.open(file.file)
    image = np.array(image)

    #image = Image.open(file.file)
    height, width, channels = image.shape
    # if the image mode is not RGB, convert it
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    encoded_image = np.expand_dims(image, axis=0)
    logger.info(f"Image original dimesions: {width}x{height}")
    # ensure our NumPy array is C-contiguous as well,
    # otherwise we won't be able to serialize it
    encoded_image = image.copy(order="C")

    k = str(uuid.uuid4())
    encoded_image = helpers.base64_encode_image(encoded_image)
    data = [{'x' : int(width/2), 'y' : int(height/2)}]
    d = {"id": k, "image": encoded_image, "height": height, "width": width, "prompts": data}
    db.rpush("image_queue_sam", json.dumps(d))

    while True:
        # attempt to grab the output predictions
        output = db.get(k)
        if output is not None:
            # add the output predictions to our data
            # dictionary so we can return it to the client
            output = output.decode("utf-8")
            # delete the result from the database and break
            # from the polling loop
            db.delete(k)
            break

        # sleep for a small amount to give the model a chance
        # to classify the input image
        time.sleep(float(os.getenv("CLIENT_SLEEP")))

    output = json.loads(output)
    indexed_scores = [(i,score) for i, (box,score) in enumerate(zip(output[0].get('box'),output[0].get('score'))) if box['w']/width > 0.7 or box['h']/height > 0.7]

    if not indexed_scores:
        logger.info('no high large areas')
        indexed_scores = [(i,score) for i, (box,score) in enumerate(zip(output[0].get('box'),output[0].get('score')))]

    largest_index, _ = max(indexed_scores, key=lambda x: x[1])
    logger.info(f"Scores: {output[0].get('score')}")
    logger.info(f"Index: {largest_index}")

    bounding_poly = np.array(output[0].get('poly')[largest_index])

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [bounding_poly], 255)
    kernel = np.ones((int(width*0.01),int(width*0.01)),np.uint8)
    #kernel = np.ones((10,10),np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations = 1)
    extracted = cv2.bitwise_and(image, image, mask=dilated_mask)
    background = np.zeros_like(image)
    background[:] = [0, 0, 0]
    background[dilated_mask == 255] = extracted[dilated_mask == 255]

    # Convert to BGR for class prediction
    image = background[:, :, ::-1].copy()
    image = np.expand_dims(image, axis=0)
    logger.info(f"Image original dimesions: {width}x{height}")
    # ensure our NumPy array is C-contiguous as well,
    # otherwise we won't be able to serialize it
    image = image.copy(order="C")

    # generate an ID for the classification then add the
    # classification ID + image to the queue
    k = str(uuid.uuid4())
    image = helpers.base64_encode_image(image.astype('float32'))
    d = {"id": k, "image": image, "height": height, "width": width}
    db.rpush(model_type, json.dumps(d))
    # keep looping until our model server returns the output
    # predictions
    data = {"success": False}
    data["segments"] = {"poly" : bounding_poly.tolist(), "box" : output[0].get('box')[largest_index]}
    while True:
        # attempt to grab the output predictions
        output = db.get(k)
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
    logger.info("made it here?")
    return_data = jsonable_encoder(data)
    # return the data dictionary as a JSON response
    return JSONResponse(content=return_data)

if __name__ == "__main__":
    print("* Starting web service...")
    app.run()
