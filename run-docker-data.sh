#!/bin/bash

#envPath="$(sourcePath=`readlink -f ${BASH_SOURCE[0]}` && echo "${sourcePath%/*}")"
scriptPath="$(sourcePath=`readlink -f "$0"` && echo "${sourcePath%/*}")"
basePath="$(cd $scriptPath && pwd)"

cd $basePath/data

docker compose up -d && \
docker compose logs -f