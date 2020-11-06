#!/bin/bash

docker build -t ga-server -f docker/server/Dockerfile .
docker run --rm -p 5000:5000  ga-server