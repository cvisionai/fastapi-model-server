FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
RUN pip3 install Pillow numpy redis python-multipart

COPY ./app /app