#!/bin/bash

#if [ -z "$1" ] 
#  then
#    echo "No folder supplied!"
#    echo "Usage: bash `basename "$0"` mot_video_folder"
#    exit
#fi

DIR="/home/cristina/Foveated-YOLT/" ; # Foveated_YOLT

# Choose set mode for caffe  [CPU | GPU]
SET_MODE=CPU

# Choose which GPU the detector runs on
GPU_ID=0

# Choose number of top prediction that you want
TOP=5

# Size of the images received by the network
SIZE_MAP=227

# Define number of kernel levels
LEVELS=5

# Define size of the fovea
#SIGMAS="0,1,10,20,30,40,50,60,70,80,90,100"
SIGMAS="70"

THRESHOLDS="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.80,0.85,0.90,0.95,1.0"
#THRESHOLDS="0.65"

# Choose method (1-CARTESIAN 2-FOVEATION 3-HYBRID)
MODE=2

# change this path to the absolute location of the network related files
FILES_FOLDER_ABSOLUTE_PATH="/home/cristina/Foveated-YOLT/files/"
MODEL_FILE="deploy_caffenet.prototxt"
WEIGHTS_FILE="bvlc_reference_caffenet.caffemodel"
MEAN_FILE="imagenet_mean.binaryproto"
LABELS_FILE="synset_words_change.txt"
DATASET="/home/cristina/Foveated-YOLT/data/images"
RESULTS_FOLDER_ABSOLUTE_PATH="/home/cristina/Foveated-YOLT/results/"
DEBUG=0
TOTAL_IMAGES=100

/home/cristina/Foveated-YOLT/build/yolt $FILES_FOLDER_ABSOLUTE_PATH $MODEL_FILE $WEIGHTS_FILE $MEAN_FILE $LABELS_FILE $DATASET $TOP $THRESHOLDS $SIZE_MAP $LEVELS $SIGMAS $RESULTS_FOLDER_ABSOLUTE_PATH $MODE $DEBUG $TOTAL_IMAGES $SET_MODE $GPU_ID




