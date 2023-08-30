#FROM cvisionai/openem_lite:latest
#RUN apt update
#RUN apt install python3.8 -y 1
#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
#RUN update-alternatives --set python /usr/bin/python3.8
#RUN pip3 install redis
#RUN pip3 install torch torchvision opencv-python

FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y python3-pip ffmpeg libgl1 libsm6 libxext6
RUN pip3 install redis
RUN pip3 install torch torchvision opencv-python

COPY ./MixOE /mixoe
COPY ./app/helpers.py /mixoe/helpers.py
WORKDIR /mixoe
RUN python3 -m pip install -r requirements.txt
COPY ./modelserv/model_mixoe/run_model_server_mixoe.py /mixoe/run_model_server_mixoe.py
COPY ./modelserv/model_mixoe/whole_fish_b7_epoch_11.pth /mixoe/model_weights.pth
CMD [ "python3", "run_model_server_mixoe.py" ]
