#!/bin/bash

VENV_DIR=${1:-.venv}

#envPath="$(sourcePath=`readlink -f ${BASH_SOURCE[0]}` && echo "${sourcePath%/*}")"
scriptPath="$(sourcePath=`readlink -f "$0"` && echo "${sourcePath%/*}")"
basePath="$(cd $scriptPath && pwd)"

cd $basePath


if [[ ! -e $VENV_DIR/pyvenv.cfg ]]; then
  echo "Creating virtual environment in $VENV_DIR..."
  python3.12 -m venv $VENV_DIR
  source $VENV_DIR/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "Using existing virtual environment in $VENV_DIR..."
  source $VENV_DIR/bin/activate
fi

python data/TCMKG.py