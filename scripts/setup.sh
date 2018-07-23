#!/bin/bash

##### Functions

function get_network_weights
{
	#download pre-trained network model
        cd files

	# Caffenet
	wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
	
	# GoogLeNet
	wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel

	# AlexNet
	wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel

	# SqueezeNet
 	wget https://github.com/DeepScale/SqueezeNet/raw/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel

	# Mobilenet
	wget https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2.caffemodel
	# VGGNet 16 weight layers
	#wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

	# Ground Truth Bounding Boxes ILSVRC 2012
	#wget https://www.dropbox.com/s/3nsb0prw9uxynkf/ILSVRC2012_bbox_val_v3.tgz?dl=0
	#wget https://raw.githubusercontent.com/Robert0812/deepsaldet/master/caffe-sal/data/ilsvrc12/imagenet_mean.binaryproto
	#wget https://git.ustclug.org/zvant/caffe-nvjetson/blob/8bc372e45125bc61896675ee2a35c674ba16362f/data/ilsvrc12/imagenet_mean.binaryproto
	#wget https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/c2c91c8e767d04621020c30ed31192724b863041/imagenet1000_clsid_to_human.txt

	wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt -O deploy_googlenet.prototxt
	wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt -O deploy_alexnet.prototxt
        wget https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/deploy.prototxt  -O deploy_squeezenet.prototxt
	wget https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2_deploy.prototxt -O deploy_mobilenet.prototxt
	cd ..
}



function get_data
{
	#sudo apt-get install unzip
	mkdir temp
        cd temp
	# get images
	#wget "https://drive.google.com/uc?export=download&id=0Bw0rlRYIVGGLb2stQ0x6VzRfcTQ" -O images.zip
	#unzip images.zip -d images

	# get results data
	#wget "https://drive.google.com/uc?export=download&id=0Bw0rlRYIVGGLTXdIdW9zLW8yTjQ" -O results.zip
	#unzip results.zip -d results

	# get ground truth
	#wget "https://drive.google.com/uc?export=download&id=0Bw0rlRYIVGGLSEFTb2F1bndUbUE" -O ground_truth.zip
	#unzip ground_truth.zip -d ground_truth
}


# Get network weights
get_network_weights

#get_data




