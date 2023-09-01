# import the necessary packages
import os
import logging
import torch
import torchvision

import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as trn
from torchvision.models import efficientnet_b7
from PIL import Image
import cv2
import argparse

import numpy as np
import helpers
import redis
import time
import json

NUM_CLASSES = 4

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# connect to Redis server
db = redis.StrictRedis(host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"), db=os.getenv("REDIS_DB"))

class DictToDotNotation:
    '''Useful class for getting dot notation access to dict'''
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

class Letterbox(object):
    def __init__(self, new_shape=(640,640)):
        assert isinstance(new_shape, (int, tuple))
        self.new_shape = new_shape

    def __call__(self,im, color=(255, 255, 255)): 
        im = np.array(im)
        shape = im.shape[:2]  # current shape [height, width], assumes HWC
        if isinstance(self.new_shape, int):
            self.new_shape = (self.new_shape, self.new_shape)

        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return Image.fromarray(im)

def load_model(args):
    '''Load model architecture and weights
    '''
    #net = efficientnet_b7(pretrained=True)
    net = efficientnet_b7(pretrained=False)
    in_features = net.classifier[1].in_features
    new_fc = nn.Linear(in_features, NUM_CLASSES)
    net.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=True),new_fc)

    pretrained_model_file = args.model_weights_file
    state_dict = torch.load(pretrained_model_file)
    net.load_state_dict(state_dict)
    net.eval()
    net.cuda()
    return net

def create_augmentations():
    '''Generate list of augmentations to perform before inference
    '''
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    preprocess = trn.Compose([
    Letterbox(528),
    trn.ToTensor(),
    trn.Normalize(mean, std)
])

    return preprocess
    
def augment_images(augs, image_batch):
    '''Generate augmented images based on list of input augmentations
    '''
    aug_imgs = []
    for img in image_batch:
        aug_imgs.append(augs(img).unsqueeze(0).cuda())
    return aug_imgs

def process_nms(model_outputs):
    '''Seperate function so that we can substitute more complex logic (e.g. inter vs intra class NMS)
    '''
    nms_op = torchvision.ops.nms
    model_outputs[0]["instances"] = model_outputs[0]["instances"][nms_op(model_outputs[0]["instances"].pred_boxes.tensor, model_outputs[0]["instances"].scores, 0.45).to("cpu").tolist()]

    return model_outputs

def process_model(model, aug_imgs):
    with torch.no_grad():
        prediction = model(aug_imgs)[0].squeeze(0).softmax(0)
        logger.info(f"Model prediction: {prediction}")
        class_ids = prediction.argmax().item()
        scores = prediction[class_ids].item()

    return prediction

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
        'Red Snapper',
        'Rose Snapper',
        'Vermillion Snapper',
        'Yellowtail Snapper']

    return label_map


def classify_process():

    import base64

    args = {}
    args['model_weights_file'] = 'model_weights.pth'
    #args["model_weights_file"] = "epoch_18.pth"
    args['score_threshold'] = 0.4
    args = DictToDotNotation(args)

    logger.info("Loading model...")
    model = load_model(args)
    logger.info("Loading complete!")

    augs = create_augmentations()
    label_map = create_labelset()
    # continually poll for new images to classify
    while True:
        # monitor queue for jobs and grab one when present
        q = db.blpop(os.getenv("IMAGE_QUEUE_MIXOE"))
        logger.info(q[0])
        q = q[1]
        imageIDs = []
        batch = None

        # deserialize the object and obtain the input image
        q = json.loads(q.decode("utf-8"))
        img_width = q["width"]
        img_height = q["height"]
        image = helpers.base64_decode_image(q["image"],
            "float32",
            (img_height, img_width,
                int(os.getenv("IMAGE_CHANS"))))

        image = image[...,::-1]
        #logger.info(f"{image[25, 25, :]}, {image[100, 100, :]}")
        logger.info(image.shape)

        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
            logger.info("converted to RGB")
        # check to see if the batch list is None. Currently
        # only batch size of 1 is supported, future growth.
        if batch is None:
            batch = [image]
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
                scores = process_model(model, aug_img_batch)
                # Process model outputs
                results = [{
                  "scores": scores.tolist()
                }]
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
