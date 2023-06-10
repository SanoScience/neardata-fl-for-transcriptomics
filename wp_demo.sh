docker compose -f ./docker/flower_tutorial/docker-compose.yml up -d data-split-service

docker compose -f ./docker/flower_tutorial/docker-compose.yml up -d server

sleep 5

docker compose -f ./docker/flower_tutorial/docker-compose.yml up -d --scale client=3