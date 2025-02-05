#!/bin/zsh

# prepare the environment with the required packages

module purge
module load Python/3.10.8-GCCcore-12.2.0

rm -rf env
virtualenv env

source env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
