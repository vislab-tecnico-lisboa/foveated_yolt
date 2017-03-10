#!/bin/bash

#if [ -z "$1" ] 
#  then
#    echo "No folder supplied!"
#    echo "Usage: bash `basename "$0"` mot_video_folder"
#    exit
#fi

DIR=$PWD; # Foveated_YOLT

# Choose set mode for caffe  [CPU | GPU]
SET_MODE=GPU

# Choose which GPU the detector runs on
GPU_ID=0

# Choose which video from the test set to start displaying
START_VIDEO_NUM=0

# Set to 0 to pause after each frame
PAUSE_VAL=1

# Choose number of top prediction that you want
TOP=5

# Size of the images received by the network
SIZE_MAP=227

# Define number of kernel levels
LEVELS=5

# Define size of the fovea
#SIGMAs={10,20,30,40,50,60,70,80,90,100,110,120,130,140}
SIGMA=50

# Choose segmentation threshold
THRESHOLD=0.75

# change this path to the absolute location of the network related files
FILES_FOLDER_ABSOLUTE_PATH=$PWD"/files/"
MODEL_FILE="deploy_caffenet.prototxt"
WEIGHTS_FILE="bvlc_reference_caffenet.caffemodel"
MEAN_FILE="imagenet_mean.binaryproto"
LABELS_FILE="synset_words_change.txt"
DATASET="/media/Data/filipa/Teste_Blur"
#DATASET="/media/Data/filipa/Blur_Validation_Set"
#BBOX=$PWD"/bbox/"
GROUND_TRUTH_LABELS=$FILES_FOLDER_ABSOLUTE_PATH"ground_truth_labels_ilsvrc12.txt"

 

echo "sigma;thres;class1;score1;x1;y1;w1;h1;class2;score2;x2;y2;w2;h2;class3;score3;x3;y3;w3;h3;class4;score4;x4;y4;w4;h4;class5;score5;x5;y5;w5;h5" > raw_bbox_parse.txt

echo "sigma;thres;class1;score1;class2;score2;class3;score3;class4;score4;class5;score5;class6;score6;class7;score7;class8;score8;class9;score9;class10;score10;class11;score11;class12;score12;class13;score13;class14;score14;class15;score15;class16;score16;class17;score17;class18;score18;class19;score19;class20;score20;class21;score21;class22;score22;class23;score23;class24;score24;class25;score25">feedback_detection_parse.txt


for SIGMA in {0,10,20,30,40,50,60,70,80,90,100,110,120}
do
    for THRESHOLD in {0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.80,0.85,0.90,0.95}
    do

        # /home/filipa/Documents/Foveated_YOLT/files/ deploy_caffenet.prototxt bvlc_caffenet.caffemodel imagenet_mean.binaryproto val.txt
        build/yolt $FILES_FOLDER_ABSOLUTE_PATH $MODEL_FILE $WEIGHTS_FILE $MEAN_FILE $LABELS_FILE $SET_MODE $GPU_ID $DATASET $TOP $THRESHOLD $SIZE_MAP $LEVELS $SIGMA $GROUND_TRUTH_LABELS
    done
done



