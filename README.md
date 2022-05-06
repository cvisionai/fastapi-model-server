
Launch on localhost using docker-compose:
```bash
cd keras-model-server-fast-api
docker network create traefik-public
docker-compose up --build
```
You can scale services to be able to handle more requests by using --scale modelserv=N --scale fast=M. Experimentally you want more web servers (fast) than model servers because the model takes a little while to process. To add a model to the running service:
```bash
docker-compose -f ./modelserv/<model_folder>/docker-compose-model-worker.yml -p model-worker-<model_name> up --build modelserv
```
NOTE: This command must be called from the base folder because of how Docker contexts and paths work.

Browser Windows:
- http://localhost:8080/dashboard
- http://localhost:8082/static/index.html

Test curl commands:
```bash
curl -k http://localhost:8082/

cd images
time curl -X POST 'http://localhost:8082/predictor/' -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@00_01_13_13.png;type=image/png"
```

Test simple-request.py
```bash
docker exec -it fast bash
cd /script/
python3 simple_request.py
```

Stress test will probably want to run outside of a container, because it will only submit to the server you're running it from. In this case, make sure the port is right.
```
python3 stress_test.py
```

### Application Organization

The structure of this application is to create a web server that handles incoming model requests, and model workers that process those requests. The link between these parts is a redis instance. Requests get routed to queues based on the desired model, and the model workers continually query the proper queue for work. Each requests is given a unique id, and this id becomes a key in redis that the model worker populates with results when finished. The web server continually monitors for that key and then populates the HTTP response with results once it is received. Practically speaking this means there is a limit on the duration that an algorithm can take before a response times out. This conops is not the most robust, but it does accommodate a large amount of use cases. Further work could extend to asynchronous processing with persistent connections or notifications of job completion, but this starts to overlap with other tools that tend to be much better that those types of tasks.

The modelserv folder is where you can add new model capabilities. The requirements for adding a new model are:

- A docker container that starts and runs the model in a loop, querying the proper queue, and posting results
- A docker compose file that adds the model to the web service
- A method for adding itself to the available model list, which is a redis key that contains a list of queues and model definitions
- A method for querying the input and return types of the model definition

Typically the docker container contains code to run an algorithm architecture, and specific model parameters such as weight files and class names need to be copied into the folder before launching/building the container. It is considered bad practice to check these into the repository, so they are usually made available elsewhere. This codebase has primarily been used for experiments with the [FathomNet](https://fathomnet.org) project, and uses examples from the [FathomNet Model Zoo](https://github.com/fathomnet/models). The model zoo has links to the required files to run some of the example architectures in this repository.

TODO

- Add score thresholding
- Configure CPU/GPU
- Configure URLs
- Configure ports
- Add health checks. Consider https://pypi.org/project/fastapi-health/