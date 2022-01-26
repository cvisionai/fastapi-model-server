# DEPRECATED. THIS IS FOR THE OLD VERSION AND LEFT FOR A REFERENCE OF HOW TO TEASE APART IN AN NGINX DEPLOYMENT


# How to Run
This app creteas a FastAPI application, which you can run using nginx, gunicorn, and uvicorn

## Requirements
Redis server - You can set this up using a Docker image. I used "docker pull redis". The command to run is `docker run --name redis-dev -d redis`.
This will run redis with port 6379 open on the docker bridge network. In order to access this without configuring your own network, you will need to find it's IP address using `docker network inspect bridge`. This value will go in settings.py.

Once you have this set up you can set up the rest of the application. First build the Dockerfile. Then you can launch the FastAPI container using
`docker run --name fast-api-test -p 8080:80 fast-api-keras-app:latest`, where port 8080 was chosen so that nginx can proxy to it on the normal port 80. 

Next build the model server container. You will need to do this from the base directory, not the model_server directory where the Dockerfile is, because of how docker context works. You can do that using the following command. `docker build -f model_server/Dockerfile -t keras-model-server .`. You can then set up the model server using the following command `docker run --gpus device=0 --rm --name model-server -d keras-model-server:latest`. The `--gpus` command is optional, if you have a gpu available on your device.

Finally, you can set up nginx to proxy this API by creating `/etc/nginx/sites-available/myprojectname` with 
```
server {
            listen       80;
            server_name  localhost;
            client_max_body_size 25M;
            location / {
                include proxy_params;
                proxy_pass http://localhost:8080;
            }
       }
```
Modify domain names and proxy information appropriately. Also modify create a symlink to `/etc/nginx/sites-enabled`.

`sudo systemctl restart nginx` should now make the project available on localhost:80. You can test the API using the python script `simple_request.py` or by using a command like `curl -X POST 'http://localhost:80/predictor/' -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@file_to_run_inference_on.png;type=image/png"`