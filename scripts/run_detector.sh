#!/bin/bash

#if [ -z "$1" ] 
#  then
#    echo "No folder supplied!"
#    echo "Usage: bash `basename "$0"` mot_video_folder"
#    exit
#fi

DIR=$PWD; # Foveated_YOLT

# Choose set mode for caffe  [CPU | GPU]
SET_MODE=CPU

# Choose which GPU the detector runs on
GPU_ID=0

# Choose which video from the test set to start displaying
START_VIDEO_NUM=0

# Set to 0 to pause after each frame
PAUSE_VAL=1

# Choose number of top prediction that you want
TOP=5

# Choose segmentation threshold
THRESHOLD=0.75

# Size of the images received by the network
SIZE_MAP=227

# Define number of kernel levels
LEVELS=5

# Define size of the fovea
SIGMA=50

# change this path to the absolute location of the network related files
FILES_FOLDER_ABSOLUTE_PATH=$PWD"/files/"
MODEL_FILE="deploy_caffenet.prototxt"
WEIGHTS_FILE="bvlc_caffenet.caffemodel"
MEAN_FILE="imagenet_mean.binaryproto"
LABELS_FILE="synset_words_change.txt"
#DATASET="/home/filipa/Documents/Validation_Set/"
DATASET="files/"


# /home/filipa/Documents/Foveated_YOLT/files/ deploy_caffenet.prototxt bvlc_caffenet.caffemodel imagenet_mean.binaryproto val.txt
build/yolt $FILES_FOLDER_ABSOLUTE_PATH $MODEL_FILE $WEIGHTS_FILE $MEAN_FILE $LABELS_FILE $SET_MODE $GPU_ID $DATASET $TOP $THRESHOLD $SIZE_MAP $LEVELS $SIGMA

#$GPU_ID
