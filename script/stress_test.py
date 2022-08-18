# USAGE
# python stress_test.py

# import the necessary packages
from threading import Thread
import requests
import time
import logging

logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# initialize the Keras REST API endpoint URL along with the input
# image path

# Note that inside the "fast" container, it uses port 80. Running
# from outside you will want to specify the right port.

MODEL_REST_API_URL = "https://your_domain_url:port/predictor/"
IMAGE_PATH = "../images/00_01_13_13.png"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {
    "file" : ("file", image, "image/png"),
    "model_type" : (None, "image_queue_yolov5")
}

# initialize the number of requests for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = 50
SLEEP_COUNT = 0.05

def call_predict_endpoint(n):
	# load the input image and construct the payload for the request

	# submit the request
	r = requests.post(MODEL_REST_API_URL, files=payload).json()

	# ensure the request was sucessful
	if r["success"]:
		logger.info("thread {} OK".format(n))

	# otherwise, the request failed
	else:
		logger.info(" thread {} FAILED".format(n))

if __name__ == "__main__":
	# loop over the number of threads
	for i in range(0, NUM_REQUESTS):
		# start a new thread to call the API
		t = Thread(target=call_predict_endpoint, args=(i,))
		t.daemon = True
		t.start()
		time.sleep(SLEEP_COUNT)

	# insert a long sleep so we can wait until the server is finished
	# processing the images
	time.sleep(60)