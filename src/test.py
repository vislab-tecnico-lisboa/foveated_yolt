import cv2
import numpy as np
import np_opencv_converter as npcv
from yolt_python import LaplacianBlending as fv
from matplotlib import pyplot as plt

img = cv2.imread('watch.jpg')

height, width, channels = img.shape

#foveated_img=fv(img,4,10)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.plot([200,300,400],[100,200,300],'c', linewidth=5)
plt.show()
