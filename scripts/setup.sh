#!/bin/bash

##### Functions

function get_network_weights
{
	mkdir files

        cd files

	# Caffenet
	wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel -O deploy_caffenet.caffemodel
	
	# GoogLeNet
	wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel -O deploy_googlenet.caffemodel
	
	# VGGNet 16 weight layers
	wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel -O deploy_vgg.caffemodel

	# AlexNet
	wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel -O deploy_alexnet.caffemodel

	# SqueezeNet
 	wget https://github.com/DeepScale/SqueezeNet/raw/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel -O deploy_squeezenet.caffemodel

	# Mobilenet
	wget https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2.caffemodel -O deploy_mobilenet.caffemodel

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




