#!/bin/bash

#envPath="$(sourcePath=`readlink -f ${BASH_SOURCE[0]}` && echo "${sourcePath%/*}")"
scriptPath="$(sourcePath=`readlink -f "$0"` && echo "${sourcePath%/*}")"
basePath="$(cd $scriptPath && pwd)"

cd $basePath


if [[ ! -e .venv ]]; then
  python3.12 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
fi

source .venv/bin/activate


python data/TCMKG.py