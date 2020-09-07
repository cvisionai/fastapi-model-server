# import the necessary packages
from keras_retinanet.models.resnet import custom_objects
import numpy as np
import settings
import helpers
import redis
import time
import json
import keras

# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST,
    port=settings.REDIS_PORT, db=settings.REDIS_DB)

def classify_process():
    SCORE_THRESHOLD = 0.4
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    MODEL_FILE = 'production_model.h5'
    print("* Loading model...")
    model = keras.models.load_model(MODEL_FILE, custom_objects=custom_objects)
    print("* Model loaded")

    # continually pool for new images to classify
    while True:
        # attempt to grab a batch of images from the database, then
        # initialize the image IDs and batch of images themselves
        queue = db.lrange(settings.IMAGE_QUEUE, 0,
            settings.BATCH_SIZE - 1)
        imageIDs = []
        batch = None

        # loop over the queue
        for q in queue:
            # deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            image = helpers.base64_decode_image(q["image"],
                settings.IMAGE_DTYPE,
                (1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH,
                    settings.IMAGE_CHANS))

            # check to see if the batch list is None
            if batch is None:
                batch = image

            # otherwise, stack the data
            else:
                batch = np.vstack([batch, image])

            # update the list of image IDs
            imageIDs.append(q["id"])

        # check to see if we need to process the batch
        if len(imageIDs) > 0:
            batch_results = []
            # classify the batch
            print("* Batch size: {}".format(batch.shape))
            _,_,dets = model.predict_on_batch(batch)

            for detections in dets:
                # classify the batch
                #print("* Batch size: {}".format(batch.shape))
                #if len(image.shape) < 4:
                #    image = np.expand_dims(image, axis=0)

                #_, _, detections = model.predict_on_batch(image)
                detections = np.expand_dims(detections,axis=0)
                # clip to image shape
                detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
                detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
                detections[:, :, 2] = np.minimum(image.shape[1], detections[:, :, 2])
                detections[:, :, 3] = np.minimum(image.shape[0], detections[:, :, 3])

                # correct boxes for image scale
                #detections[0, :, :4] /= scale

                # change to (x, y, w, h) (MS COCO standard)
                detections[:, :, 2] -= detections[:, :, 0]
                detections[:, :, 3] -= detections[:, :, 1]

                results = []
                # compute predicted labels and scores
                for detection in detections[0, ...]:
                    label = np.argmax(detection[4:])
                    # append detections for each positively labeled class
                    if float(detection[4 + label]) > SCORE_THRESHOLD:
                        image_result = {
                            'category_id' : str(label),
                            'scores'      : [str(det) for i,det in 
                                            enumerate(detection) if i >= 4],
                            'bbox'        : str((detection[:4]).tolist()),
                        }
                        # append detection to results
                        results.append(image_result)
                batch_results.append(results)
                # loop over the image IDs and their corresponding set of
                # results from our model
            # loop over the image IDs and their corresponding set of
            # results from our model
            for (imageID, resultSet) in zip(imageIDs, batch_results):
                # store the output predictions in the database, using
                # the image ID as the key so we can fetch the results
                db.set(imageID, json.dumps(resultSet))

            # remove the set of images from our queue
            db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)

        # sleep for a small amount
        time.sleep(settings.SERVER_SLEEP)

# if this is the main thread of execution start the model server
# process
if __name__ == "__main__":
    classify_process()
