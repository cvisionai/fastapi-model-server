FROM cvisionai/openem_lite:latest

RUN apt update
RUN pip3 install redis 
COPY ./modelserv/run_model_server.py /scripts/run_model_server.py
COPY ./app/settings.py /scripts/settings.py
COPY ./app/helpers.py /scripts/helpers.py
COPY production_model.h5 /scripts/production_model.h5

WORKDIR /scripts

CMD [ "python3", "run_model_server.py" ]
