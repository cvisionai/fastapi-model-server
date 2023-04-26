FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10
RUN apt-get update && apt-get install --no-install-recommends -y libgl1 python3-opencv python-dev-is-python3 && rm -fr /var/lib/apt/lists/*
RUN pip3 install Pillow numpy redis python-multipart requests aiofiles opencv-python

RUN mkdir -p /app/app
COPY . /app/app
