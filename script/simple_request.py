# USAGE
# python simple_request.py

# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost/predictor/"
IMAGE_PATH = "../images/00_01_13_13.png"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"file": image, "type": "image/png"}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was sucessful
if r["success"]:
	# loop over the predictions and display them
	for (i, result) in enumerate(r["predictions"]):
		print(f"{i + 1}. {result}")

# otherwise, the request failed
else:
	print("Request failed")