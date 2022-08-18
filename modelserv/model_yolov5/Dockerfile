FROM cvisionai/openem_lite:latest
RUN apt update
# Install python 3.7
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.7 -y
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN update-alternatives --set python /usr/bin/python3.7
RUN pip3 install redis
RUN pip3 install torch torchvision torchaudio
# WARNING: You will need to clone yolov5 repo and change path accordingly

COPY ./modelserv/model_yolov5/run_model_server_yolov5.py /scripts/run_model_server_yolov5.py
COPY ./app/helpers.py /scripts/helpers.py
COPY ./modelserv/model_yolov5/mbari-mb-benthic-33k.pt /scripts/mbari-mb-benthic-33k.pt

COPY ./yolov5 /yolov5
WORKDIR "/yolov5"
RUN pip3 install -r requirements.txt

WORKDIR /scripts

CMD [ "python3", "run_model_server_yolov5.py" ]
