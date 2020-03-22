#!/bin/bash

# Inputs:
# SCRIPT: ${1} -- the gan.py script
# IMAGEDIR: ${2} -- the directory to save generated images to (does not have to exist)
# MODELDIR: ${3} -- the directory to save the models to (does not have to exist)

SCRIPT=${1}
IMAGEDIR=${2}
MODELDIR=${3}
EPOCHSIZE=8192
NUMEPOCHS=10

for NUMBER in {0..9}
do
    echo "Training GAN to create number ${NUMBER}"
    python $SCRIPT $NUMBER $NUMEPOCHS --epoch-size $EPOCHSIZE -i $IMAGEDIR -m $MODELDIR
    if [ $? -ne 0 ]
    then
	    echo "Something went wrong generating number $NUMBER"
	    exit 1
    fi
done
exit 0
