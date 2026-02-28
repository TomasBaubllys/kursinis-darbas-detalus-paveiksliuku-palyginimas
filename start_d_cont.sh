#!/bin/bash

# This script starts a docker container for testing

IMAGE_NAME=reid-project

if  ! "$(docker inspect ${IMAGE_NAME})" >/dev/null 2>&1 ; then
	docker buildx build --load -t $IMAGE_NAME .
fi

docker run --gpus all -it --rm --shm-size=8gb ${IMAGE_NAME}

