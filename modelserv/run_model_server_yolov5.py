# import the necessary packages
import torchvision
import os
import torch
import logging

import numpy as np
import settings
import helpers
import redis
import time
import json
import random

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST,
    port=settings.REDIS_PORT, db=settings.REDIS_DB)

class DictToDotNotation:
    '''Useful class for getting dot notation access to dict'''
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

def parse_args():
    parser = configargparse.ArgumentParser(description="Testing script for testing video data.",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add("-c", "--config", required=True, is_config_file=True, help="Default config file path")
    parser.add("--model_config_file", help="Detectron2 config file")
    parser.add("--model_weights_file", help=".pth file with Detectron2 weights")
    parser.add("--score_threshold", help="Score threshold for boxes", type=float, default=0.4)

    return parser.parse_args()

def load_model(args):
    '''Load model architecture and weights
    '''
    model = torch.hub.load('ultralytics/yolov5','custom',path=args.model_weights_file)
    model.conf = args.score_threshold

    return model

def create_augmentations():
    '''Generate list of augmentations to perform before inference
    '''
    aug1 = T.ResizeShortestEdge(
            [720, 720], 1280)
    aug2 = T.ResizeShortestEdge(
            [540, 540], 960)
    aug3 = T.ResizeShortestEdge(
            [1080, 1080], 1920)

    return [aug1, aug2, aug3]
    
def augment_images(augs, image_batch):
    '''Generate augmented images based on list of input augmentations
    '''
    if len(image_batch) > 0:
        aug_imgs = []
        for img in image_batch:
            aug_imgs.append({f"im_{i}" : augs[i].get_transform(img).apply_image(img) for i in range(len(augs))})
    return aug_imgs

def process_nms(model_outputs):
    '''Seperate function so that we can substitute more complex logic (e.g. inter vs intra class NMS)
    '''
    nms_op = torchvision.ops.nms
    model_outputs[0]["instances"] = model_outputs[0]["instances"][nms_op(model_outputs[0]["instances"].pred_boxes.tensor, model_outputs[0]["instances"].scores, 0.45).to("cpu").tolist()]

    return model_outputs

def process_model(model, aug_imgs, im_width, im_height):
    with torch.no_grad():
        model_outputs = model(aug_imgs[0], im_width)

    return model_outputs

def process_model_outputs(model_outputs):
    '''Transform model outputs into lists of boxes, scores, and labels
    '''
    
    results = model_outputs.pred[0].cpu().numpy()
    output_boxes = [[float(arr[0]),float(arr[1]),float(arr[2]),float(arr[3])] for arr in results]
    output_scores = [[float(arr[4])] for arr in results]
    output_classes = [int(arr[5]) for arr in results]

    return output_boxes, output_scores, output_classes

def create_labelset(model):
    '''Generate map between class numbers and labels from model
    '''
    label_map=model.names

    return label_map

def classify_process():

    args = {}
    args['model_weights_file'] = '/scripts/mbari-mb-benthic-33k.pt'
    args['score_threshold'] = 0.1
    args = DictToDotNotation(args)

    logger.info("Loading model...")
    model = load_model(args)
    logger.info("Loading complete!")

    label_map = create_labelset(model)
    # continually pool for new images to classify
    while True:
        # attempt to grab a batch of images from the database, then
        # initialize the image IDs and batch of images themselves
        queue = db.lrange(settings.IMAGE_QUEUE, 0,
            settings.BATCH_SIZE - 1)

        # remove the set of images from our queue
        db.ltrim(settings.IMAGE_QUEUE, len(queue), -1)
        imageIDs = []
        batch = None

        # loop over the queue
        for q in queue:
            # deserialize the object and obtain the input image
            q = json.loads(q.decode("utf-8"))
            img_width = q["width"]
            img_height = q["height"]
            image = helpers.base64_decode_image(q["image"],
                settings.IMAGE_DTYPE,
                (1, img_height, img_width,
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
            logger.info(imageIDs)
            batch_results = []

            #im_height,im_width,_ = frame.shape

            # Run model
            logger.info(f"Running model")
            model_outputs = process_model(model, batch, img_width, img_height)
            # Process model outputs
            logger.info(f"Processing outputs")
            output_boxes, output_scores, output_classes = process_model_outputs(model_outputs)

            results = []
            for box,scores,label in zip(output_boxes, output_scores, output_classes):
                image_result = {
                        'category_id' : label_map[label],
                            'scores'      : [scores],
                            'bbox'        : box,
                        }
            
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

        # sleep for a small amount
        time.sleep(settings.SERVER_SLEEP + random.uniform(0.05,0.1))

# if this is the main thread of execution start the model server
# process
if __name__ == "__main__":
    classify_process()
