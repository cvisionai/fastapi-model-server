
Launch on localhost using docker-compose:
```bash
cd keras-model-server-fast-api
docker network create traefik-public
docker-compose up --build
```

Browser Windows:
- http://localhost:8080/dashboard
- http://fast.localhost

Test curl commands:
```bash
curl -k http://fast.localhost/

cd images
time curl -X POST 'http://fast.localhost/predictor/' -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@demo-face.jpeg;type=image/png"
```

Test simple-request.py
```bash
docker exec -it fast bash
cd /script/
python3 simple_request.py
python3 stress_test.py
```