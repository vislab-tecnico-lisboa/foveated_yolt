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
	
	# VGGNet 16 weight layers
	wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

	# AlexNet
	wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel

	# Ground Truth Bounding Boxes ILSVRC 2012
	wget https://www.dropbox.com/s/3nsb0prw9uxynkf/ILSVRC2012_bbox_val_v3.tgz?dl=0
	
	cd ..
}



# Get videos
get_network_weights


