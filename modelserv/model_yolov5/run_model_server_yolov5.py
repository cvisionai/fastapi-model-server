# import the necessary packages
import os
import torch
import logging

import numpy as np
import helpers
import redis
import time
import json
import random

import sys
from pathlib import Path

import numpy as np

ROOT = '/work/yolov5' # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, cv2, non_max_suppression, scale_boxes
from utils.torch_utils import select_device


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

def load_image(q, stride, auto):
    """
    Load an image.
    """
    # TODO: Use image_info instead of path
    # deserialize the object and obtain the input image
    q = json.loads(q.decode("utf-8"))
    img_width = q["width"]
    img_height = q["height"]
    img0 = helpers.base64_decode_image(q["image"],
        os.getenv("IMAGE_DTYPE"),
        (1, img_height, img_width,
            int(os.getenv("IMAGE_CHANS"))))

    img0 = np.squeeze(img0)

    logger.info(f"Image shape: {img0.shape}")
    assert img0 is not None, f'Image Not Found'

    # Padded resize
    if img_height > 960:
        img = letterbox(img0, (1280, 1280), stride=stride, auto=auto)[0]
    else:
        img = letterbox(img0, (640, 640), stride=stride, auto=auto)[0]

    # Convert
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)
    id = q["id"]

    return img, img0, id


def get_image_info_blocking():
    """
    Get info of an image to process. Block until available.
    """
    q = db.blpop(os.getenv("IMAGE_QUEUE_YOLOV5"))
    logger.info(q[0])
    q = q[1]
    return q

def load_model(args):
    '''Load model architecture and weights
    '''
    model = torch.hub.load('ultralytics/yolov5','custom',path=args.model_weights_file)
    model.conf = args.score_threshold
    model.agnostic = True

    return model

def create_augmentations():
    '''Generate list of augmentations to perform before inference
    '''
    pass
    
def augment_images(augs, image_batch):
    '''Generate augmented images based on list of input augmentations
    '''
    pass

def process_nms(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=1000):
    '''Seperate function so that we can substitute more complex logic (e.g. inter vs intra class NMS)
    '''
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    return pred

def process_model(model, im, augment, device):
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    
    logger.info(f"Batch shape: {im.shape}")
    # Inference
    pred = model(im, augment=augment, visualize=False)

    return pred

def process_model_outputs(model_outputs, im, im0s):
    '''Transform model outputs into lists of boxes, scores, and labels
    '''
    output_boxes = []
    output_scores = []
    output_classes = []
    # Process predictions
    for det in model_outputs:  # per image
        #seen += 1
        det = det.cpu().numpy()
        im0 = im0s.copy()
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[1:], det[:, :4], im0.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            output_boxes.append([float(elem) for elem in xyxy])
            output_scores.append(float(conf))
            output_classes.append(cls)

    return output_boxes, output_scores, output_classes

def create_labelset(model):
    '''Generate map between class numbers and labels from model
    '''
    label_map=model.names

    return label_map

@torch.no_grad()
def classify_process(
    weights='/scripts/mbari-mb-benthic-33k.pt',  # model.pt path(s)
    data='mbari_classes.yaml',  # dataset.yaml path
    conf_thres=0.1,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='0', # Set to CPU if no GPU is present
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=True,  # class-agnostic NMS
    augment=False,  # augmented inference
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    imgsz=[720,1280]
):
    # Load model
    device = select_device(device)
    logger.info("Loading model...")
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    logger.info("Loading complete!")
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    label_map = create_labelset(model)

    # Dataloader
    bs = 1  # batch_size
    
    # Run inference loop
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen = 0
    # continually pool for new images to classify
    while True:
        try:
            # monitor queue for jobs and grab one when present
            q = get_image_info_blocking()
            imageIDs = []
            batch = None

            im, im0s,id = load_image(q, stride, auto=False)
            # check to see if the batch list is None
            if batch is None:
                batch = im
            # otherwise, stack the data
            else:
                batch = np.vstack([batch, im])

            # update the list of image IDs
            imageIDs.append(id)

            # check to see if we need to process the batch
            if len(imageIDs) > 0:
                logger.info(imageIDs)
                batch_results = []

                #im_height,im_width,_ = frame.shape

                # Run model
                logger.info(f"Running model")
                model_outputs = process_model(model, np.expand_dims(im,0), augment, device)
                model_outputs = process_nms(model_outputs, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                # Process model outputs
                logger.info(f"Processing outputs")
                output_boxes, output_scores, output_classes = process_model_outputs(model_outputs,im, im0s)

                results = []
                for box,scores,label in zip(output_boxes, output_scores, output_classes):
                    logger.info(f"ID: {label}, scores: {scores}, box: {box}")
                    image_result = {
                            'category_id' : label_map[label],
                                'scores'      : [float(scores)],
                                'bbox'        : [float(elem) for elem in box],
                            }
                
                    results.append(image_result)
                batch_results.append(results)
                # loop over the image IDs and their corresponding set of
                # results from our model
                for (imageID, resultSet) in zip(imageIDs, batch_results):
                    # store the output predictions in the database, using
                    # the image ID as the key so we can fetch the results
                    db.set(imageID, json.dumps(resultSet))
        except KeyboardInterrupt:
            logger.info("Stopped.")


# if this is the main thread of execution start the model server
# process
if __name__ == "__main__":
    classify_process()
