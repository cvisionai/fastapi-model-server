# USAGE
# python simple_request.py

# import the necessary packages
import requests, time

# initialize the Keras REST API endpoint URL along with the input
# image path
MODEL_REST_API_URL = "https://adamant.tator.io:8082/predictor/"
#MODEL_REST_API_URL = "https://your_domain_url:port/predictor/"
IMAGE_PATH = "../images/00_01_13_13.png"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {
    "file" : ("file", image, "image/png"),
    "model_type" : (None, "image_queue_detectron2")
}
start_time = time.time()
# submit the request
r = requests.post(MODEL_REST_API_URL, files=payload).json()
# ensure the request was sucessful
print(f"Request took {time.time() - start_time}")
if r["success"]:
	# loop over the predictions and display them
	for (i, result) in enumerate(r["predictions"]):
		print(f"{i + 1}. {result}")

# otherwise, the request failed
else:
	print("Request failed")


MODEL_REST_API_URL = "https://adamant.tator.io:8082/sam/"
payload = {
	"file" : ("file", image, "image/png"),
	"prompt_list" : [
	{"x" : 252,
     "y" : 782}
]
}
data = [
	{"x" : 252,
     "y" : 782}
]

r = requests.post(MODEL_REST_API_URL, files=payload).json()