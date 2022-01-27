
Launch on localhost using docker-compose:
```bash
cd keras-model-server-fast-api
docker network create traefik-public
docker-compose up --build
```
You can scale services to be able to handle more requests by using --scale modelserv=N --scale fast=M. Experimentally you want more web servers (fast) than model servers because the model takes a little while to process

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

TODO

- Add box overlay capability
- Add score thresholding
- Configure CPU/GPU
- Configure modelserver type
- Configure TLS
- Configure URLs
- Configure ports
- Add health checks. Consider https://pypi.org/project/fastapi-health/