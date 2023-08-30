#FROM cvisionai/openem_lite:latest
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

RUN apt update
RUN apt install -y python3-pip git
RUN pip3 install redis configargparse
RUN pip3 install torch torchvision torchaudio
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
#RUN pip3 install detectron2 -f \
#  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
#RUN pip3 install detectron2 -f \
#  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
COPY ./modelserv/model_detectron2/run_model_server_detectron.py /scripts/run_model_server_detectron.py
#RUN mkdir -p /scripts
COPY ./app/helpers.py /scripts/helpers.py
COPY ./modelserv/model_detectron2/model_final.pth /scripts/model_final.pth
COPY ./modelserv/model_detectron2/fathomnet_config_v2_1280.yaml /scripts/fathomnet_config_v2_1280.yaml

WORKDIR /scripts

CMD [ "python3", "run_model_server_detectron.py" ]
