# import the necessary packages
import torchvision
import os
import torch
import logging

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.data import transforms as T
from detectron2.checkpoint import DetectionCheckpointer

import configargparse
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
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(args.model_config_file)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.score_threshold
    cfg.MODEL.WEIGHTS = os.path.join(args.model_weights_file)  # path to the model we just trained
    model = build_model(cfg)  # returns a torch.nn.Module
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()
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
        model_inputs = []
        for _,value in aug_imgs.items():
            model_inputs.append({
                "image" : torch.as_tensor(value.astype("float32").transpose(2,0,1)),
                "height" : im_height,
                "width" : im_width
            })
        model_outputs = model(model_inputs)

    return model_outputs

def process_model_outputs(model_outputs):
    '''Transform model outputs into lists of boxes, scores, and labels
    '''
    # Gather boxes, scores, and labels from each augmented run
    all_model_boxes = torch.cat(tuple([model_output.get("instances").pred_boxes.tensor for model_output in model_outputs]), 0)
    all_model_scores = torch.cat(tuple([model_output.get("instances").scores for model_output in model_outputs]),0)
    all_model_classes = torch.cat(tuple([model_output.get("instances").pred_classes for model_output in model_outputs]),0)

    # Assign all boxes to a single Instances object
    model_outputs[0]["instances"].pred_boxes.tensor = all_model_boxes
    model_outputs[0]["instances"].scores = all_model_scores
    model_outputs[0]["instances"].pred_classes = all_model_classes

    # Perform NMS on combined images
    model_outputs = process_nms(model_outputs)

    frame_boxes = model_outputs[0]["instances"].to("cpu")
    output_boxes = frame_boxes.pred_boxes.tensor.tolist()
    output_scores = frame_boxes.scores.tolist()
    output_classes = frame_boxes.pred_classes.tolist()

    return output_boxes, output_scores, output_classes

def create_labelset():
    '''Generate map between class numbers and labels from model
    '''
    label_map=[
        'Anemone',
        'Fish',
        'Eel',
        'Gastropod',
        'Sea star',
        'Feather star',
        'Sea cucumber',
        'Urchin',
        'Glass sponge',
        'Sea fan',
        'Soft coral',
        'Sea pen',
        'Stony coral',
        'Ray',
        'Crab',
        'Shrimp',
        'Squat lobster',
        'Flatfish',
        'Sea spider',
        'Worm']

    return label_map


def classify_process():

    args = {}
    args['model_config_file'] = 'fathomnet_config_v2_1280.yaml'
    args['model_weights_file'] = 'model_final.pth'
    args['score_threshold'] = 0.2
    args = DictToDotNotation(args)

    logger.info("Loading model...")
    model = load_model(args)
    logger.info("Loading complete!")

    augs = create_augmentations()
    label_map = create_labelset()
    # continually pool for new images to classify
    while True:
        # monitor queue for jobs and grab one when present
        q = db.blpop(settings.IMAGE_QUEUE)
        logger.info(q[0])
        q = q[1]
        imageIDs = []
        batch = None

        # deserialize the object and obtain the input image
        q = json.loads(q.decode("utf-8"))
        img_width = q["width"]
        img_height = q["height"]
        image = helpers.base64_decode_image(q["image"],
            settings.IMAGE_DTYPE,
            (1, img_height, img_width,
                settings.IMAGE_CHANS))

        # check to see if the batch list is None. Currently
        # only batch size of 1 is supported, future growth.
        if batch is None:
            batch = image
        # otherwise, stack the data
        else:
            batch = np.vstack([batch, image])

        # update the list of image IDs
        imageIDs.append(q["id"])

        # check to see if we need to process the batch. 
        # Currently only batch size of 1 is supporeted,
        # future growth.
        if len(imageIDs) > 0:
            logger.info(imageIDs)
            batch_results = []
            # Create augmented images based on list of augmentations
            logger.info(f"Augmenting {len(batch)} images")
            aug_imgs = augment_images(augs, batch)
            # Run model
            for aug_img_batch in aug_imgs:
                logger.info(f"Running model")
                model_outputs = process_model(model, aug_img_batch, img_width, img_height)
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
            for (imageID, resultSet) in zip(imageIDs, batch_results):
                # store the output predictions in the database, using
                # the image ID as the key so we can fetch the results
                db.set(imageID, json.dumps(resultSet))


# if this is the main thread of execution start the model server
# process
if __name__ == "__main__":
    classify_process()
