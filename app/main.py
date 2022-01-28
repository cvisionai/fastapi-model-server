# import the necessary packages
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from PIL import Image
import numpy as np
import app.settings as settings
import app.helpers as helpers
import redis
import uuid
import time
import json
import io
import logging

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
    "http://fast.localhost:8080",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="/static-files"), name="static")

db = redis.StrictRedis(host=settings.REDIS_HOST,
    port=settings.REDIS_PORT, db=settings.REDIS_DB)
db.ping() 

print(f"connected to redis: {settings.REDIS_HOST}") 

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

@app.post("/predictor/")
def predict(file: UploadFile = File(...)):
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    # read the image in PIL format and prepare it for
    # classification
    #image = flask.request.files["image"].read()
    #image = Image.open(io.BytesIO(file.file))
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
    #image = prepare_image(image,
    #    (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))
    # ensure our NumPy array is C-contiguous as well,
    # otherwise we won't be able to serialize it
    image = image.copy(order="C")

    # generate an ID for the classification then add the
    # classification ID + image to the queue
    k = str(uuid.uuid4())
    image = helpers.base64_encode_image(image)
    d = {"id": k, "image": image, "height": height, "width": width}
    db.rpush(settings.IMAGE_QUEUE, json.dumps(d))
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
        time.sleep(settings.CLIENT_SLEEP)

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
