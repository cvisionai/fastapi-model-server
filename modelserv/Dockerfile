FROM cvisionai/openem_lite:latest
RUN apt update
RUN pip3 install redis configargparse
RUN pip3 install torch torchvision torchaudio
RUN pip3 install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
COPY ./modelserv/run_model_server_detectron.py /scripts/run_model_server_detectron.py
COPY ./app/settings.py /scripts/settings.py
COPY ./app/helpers.py /scripts/helpers.py
COPY ./model_final.pth /scripts/model_final.pth
COPY ./fathomnet_config_v2_1280.yaml /scripts/fathomnet_config_v2_1280.yaml

WORKDIR /scripts

CMD [ "python3", "run_model_server_detectron.py" ]
