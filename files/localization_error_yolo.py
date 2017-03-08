import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL
from PIL import Image
import caffe
import cv2

from scipy import ndimage
from skimage.filter import threshold_otsu
import Image
from numpy import array, argwhere
import matplotlib.patches as patches
from glob import glob
from thesis_functions import *
import matplotlib.pyplot as mplot

#caffe.set_mode_cpu()        # mode

caffe.set_mode_gpu()
caffe.set_device(0)

#########################################################################################
# GLOBAL VARIABLES
imagenetModelFile = 'deploy_' + sys.argv[1] + '.prototxt'
imagenetTrainedModel = 'bvlc_' + sys.argv[1] + '.caffemodel'

imagenetMeanFile = '/home/filipa/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
LABELS_IMAGENET = '/home/filipa/PycharmProjects/Proposal_Code/synset_words_change.txt'
#IMAGE_PATH_FOLDER = '/media/Data/filipa/ILSVRC2012_img_val/'
IMAGE_PATH_FOLDER = '/media/Data/filipa/Validation_Part_B11/'
#IMAGE_PATH_FOLDER = '/media/Data/filipa/Blur_Validation_Set/'  # Blur Validation Set
IMAGE_FOLDER = 'Actual_Results/'
#LABELS_IMAGENET = '/home/filipa/PycharmProjects/Proposal_Code/Labels/val.txt'

#########################################################################################


#########################################################################################
#                                                                                       #
#                                   MAIN PROGRAM                                        #
#                                                                                       #
#########################################################################################
#
# Initialization
localization_error = []
# Matriz de 20 por 50000
#classes_bbox = [[0 for x in range(20)] for y in range(50000)]

# Create network for Test
net = caffe.Net(imagenetModelFile,     # arquitecture
                imagenetTrainedModel,  # weights
                caffe.TEST)            # phase

# Configure Preprocessing
transformer = config_preprocess(net, imagenetMeanFile)

# For each image
for path in sorted(glob(IMAGE_PATH_FOLDER + "ILSVRC2012_val_*.JPEG")):

    m = path.rstrip('.JPEG')[-5:]
    print(m)
    # Initialization
    final_top_5 = []  # final top 5 classes for each image
    bbox = [[0 for x in range(4)] for y in range(5)]
    #BBOX_IMAGENET = '/media/Data/filipa/ILSVRC2012_img_val_bbox/ILSVRC2012_val_000' + str(m) + '.xml'

    # Load image
    im = caffe.io.load_image(path)

    # perform the preprocessing we've set up
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    #print("faz forward\n")
    # Compute forward
    out = net.forward()

    #print '\n Predicted class is (class index):', out['prob'].argmax()

    # Predict first top 5 labels
    top_list, top_k, labels = predicted_first_top_labels(LABELS_IMAGENET, net, out)
    #print top_list

    # for each class
    for k in range(0, 5):

        #########################################################################################
        #########################################################################################
        #                       Weakly Supervised Object Localisation                           #
        #        Class Saliency Extraction + Segmentation Mask + Bounding Box to locate         #
        #########################################################################################


        # Get Saliency map for a given class
        saliency = get_saliency_map(net, top_k, k)

        # Get Segmentation Mask
        #bbox = segmentation_mask(saliency, k, bbox)

        # Make copy of saliency map
        foreground_mask = np.array(saliency)

        # Mask with top ranked pixels
        for i in range(0, 227):
            for j in range(0, 227):
                if foreground_mask[i][j] < 9.0e-01:
                    foreground_mask[i][j] = 0

        # Find nonzero elements and set boundaries
        B = argwhere(foreground_mask)
        (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1

        # Store points to plot final bounding boxes
        bbox[k][0] = xstart
        bbox[k][1] = xstop
        bbox[k][2] = ystart
        bbox[k][3] = ystop

        #print("bbox da classe " + str(k) + "\n")


    ######################################################################
    # Read all ground truth bboxes of an image                           #
    # Calculate overlap percentage of ground truth bbox with 5 predicted #
    # bboxes by the network                                              #
    ######################################################################
    # localization_error = get_localization(BBOX_IMAGENET, bbox, path, m, localization_error)

    BBOX_IMAGENET = '/media/Data/filipa/ILSVRC2012_img_val_bbox/ILSVRC2012_val_000' + str(m) + '.xml'


    file = minidom.parse(BBOX_IMAGENET)

    sizes = file.getElementsByTagName("size")
    for size in sizes:
        image_width = size.getElementsByTagName("width")[0].firstChild.data
        image_height = size.getElementsByTagName("height")[0].firstChild.data

    j = 0
    overlap_list = []

    bboxes = file.getElementsByTagName("bndbox")
    for bndbox in bboxes:
        xmin = bndbox.getElementsByTagName("xmin")[0].firstChild.data
        ymin = bndbox.getElementsByTagName("ymin")[0].firstChild.data
        xmax = bndbox.getElementsByTagName("xmax")[0].firstChild.data
        ymax = bndbox.getElementsByTagName("ymax")[0].firstChild.data

        im = Image.open(path)
        im = im.resize((227, 227), PIL.Image.ANTIALIAS)

        fig, ax = plt.subplots(1)  # Create figure and axes
        ax.imshow(im)

        # Create a Rectangle patch
        rect = patches.Rectangle(
            (float(227 * int(xmin) / int(image_width)), float(227 * int(ymin) / int(image_height))),
            float(227 * int(xmax) / int(image_width)) - float(227 * int(xmin) / int(image_width)),
            float(227 * int(ymax) / int(image_height)) - float(227 * int(ymin) / int(image_height)),
            linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Have number of ground truth bbox per image
        j += 1
        # plt.savefig('ground_local_resized_' + str(m) + '_' + str(j) + '.png')
        #print("le ground truth bbox\n")

        #####################################################################################
        #      Check if top-1 class bbox overlap at least 50% with the ground truth bbox    #
        #                                                                                   #
        #####################################################################################
        x_g = float(227 * int(xmin) / int(image_width))
        y_g = float(227 * int(ymin) / int(image_height))
        width_g = float(227 * int(xmax) / int(image_width)) - float(227 * int(xmin) / int(image_width))
        height_g = float(227 * int(ymax) / int(image_height)) - float(227 * int(ymin) / int(image_height))
        ground_truth_bbox = [x_g, y_g, width_g, height_g]

        for k in range(0, 5):
            x_p = bbox[k][0]  # xstart
            y_p = bbox[k][2]  # ystart
            width_p = bbox[k][1] - bbox[k][0]  # xstop - xstart
            height_p = bbox[k][3] - bbox[k][2] # ystop - ystart
            predicted_bbox = [x_p, y_p, width_p, height_p]

            # # Function which calculates area of intersection of two rectangles.
            intersectionArea = max(0, min(float(227 * int(xmax) / int(image_width)),
                                          bbox[k][1]) - max(x_g, x_p)) * max(
                0, min(float(227 * int(ymax) / int(image_height)), bbox[k][3]) - max(y_g, y_p))

            # print intersectionArea
            unionCoords = [min(x_g, x_p), min(y_g, y_p), max(x_g + width_g - 1, x_p + width_p - 1),
                           max(y_g + height_g - 1, y_p + height_p - 1)]

            unionArea = (unionCoords[2] - unionCoords[0] + 1) * (unionCoords[3] - unionCoords[1] + 1)
            # print unionArea
            overlapArea = intersectionArea / unionArea  # This should be greater than 0.5 to consider it as a valid detection.

            # print overlapArea*100
            overlap_list.append(overlapArea * 100)

    # print overlap_list

    #print("antes do append\n")
    # Calculate localization error
    if any(i >= 50 for i in overlap_list):
        # print 'maior'
        localization_error.append(0)
    else:
        localization_error.append(1)




# Calculate localization error
#"calculate_localization_error(localization_error)
#np.savetxt('local_error_19000_yolo.txt', localization_error)

local_error = (sum(i for i in localization_error) / float(len(localization_error))) * float(100.0)

print 'Localization Error [YOLO-Google-Blur-Segment 9.0]: ', local_error

#local_file = open('localization_results_yolo.txt', 'w')
#local_file.write('Caffenet\n')
#local_file.write(localization_error)
#local_file.write(local_error)
#local_file.close()



