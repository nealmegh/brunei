#!/usr/bin/env bash
# exit on error
set -o errexit

python3 -m pip install --upgrade pip

pip3 install -r requirements.txt
