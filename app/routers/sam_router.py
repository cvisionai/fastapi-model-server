from fastapi import APIRouter, File, UploadFile, Form, Response, Depends
from pydantic import BaseModel, parse_obj_as
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter
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
from ..database import get_db

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/sam-embed/")
def predict_sam_embed(db: redis.Redis = Depends(get_db), file: UploadFile = File(...)):
    contents = file.file.read()
    k = str(uuid.uuid4())
    k_img = str(uuid.uuid4())
    db.set(k_img, contents)
    d = {"id": k, "img_key" : k_img, "prompts": {"fake" : "data"}, "embedding" : True}
    db.rpush("image_queue_sam", json.dumps(d))

    while True:
        # attempt to grab the output predictions
        output = db.get(k)
        if output is not None:
            # delete the result from the database and break
            # from the polling loop
            db.delete(k)
            db.delete(k_img)
            break

        # sleep for a small amount to give the model a chance
        # to classify the input image
        time.sleep(float(os.getenv("CLIENT_SLEEP")))
    return Response(content=output, media_type="application/octet-stream")

@router.post("/sam_v2/")
def predict_sam(db: redis.Redis = Depends(get_db), data: str = Form(...), file: UploadFile = File(...)):
    
    logger.info(data)
    return_data = {"success": False}
    logger.info(f"Prompts: {data}")
    data = json.loads(data)
    contents = file.file.read()
    k = str(uuid.uuid4())
    k_img = str(uuid.uuid4())
    db.set(k_img, contents)
    d = {"id": k, "img_key" : k_img, "prompts": data}
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

@router.post("/sam_multi_v2/")
def predict_sam_multi(db: redis.Redis = Depends(get_db), data: str = Form(...), file: UploadFile = File(...)):

    logger.info(data)
    return_data = {"success": False}
    logger.info(f"Prompts: {data}")
    data = json.loads(data)
    contents = file.file.read()
    k = str(uuid.uuid4())
    k_img = str(uuid.uuid4())
    db.set(k_img, contents)
    d = {"id": k, "img_key" : k_img, "prompts": data}
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