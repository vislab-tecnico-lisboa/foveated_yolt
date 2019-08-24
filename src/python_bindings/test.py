import cv2
import numpy as np
import pysmooth_foveation as fv
from matplotlib import pyplot as plt
from random import randint
import time

rho=-0.5

# FOVEA SIZE
sigma_xx=50
sigma_yy=50
#sigma_xy=int(np.floor(rho*sigma_xx*sigma_yy))
sigma_xy=0

#PYRAMID LEVELS
levels=5
img = cv2.imread('images/pedestrian.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height, width, channels = img.shape

# Create the Laplacian blending object
my_lap_obj=fv.LaplacianBlending(width,height,levels,sigma_xx,sigma_yy,sigma_xy)
try:
    while True:
        start = time.time()

        sigma_x=sigma_xx
        sigma_y=sigma_yy

        #sigma_x=randint(1, sigma_xx)
        #sigma_y=randint(1, sigma_yy)

        center=np.array([randint(1,width), randint(1,height)])         # RANDOM FIXATION POINTS
        #center=[int(width/2.0), int(height/2.0)]

        # FOVEA SIZE
        my_lap_obj.update_fovea(width,height,sigma_x,sigma_y,sigma_xy)

        #print npcv.test_np_mat(np.array(center))
        # Foveate the image
        #print npcv.test_np_mat(np.array(center))
        #print img.depth()
        foveated_img=my_lap_obj.Foveate(img,center)
        #foveated_img = foveated_img.astype(int)
        #foveated_img=cv2.cvtColor(foveated_img, cv2.COLOR_BGR2RGB)

        cv2.imshow('image',foveated_img)
        cv2.waitKey(10)
	#cv2.destroyAllWindows()

        end = time.time()
        #print 'elapsed time:'+str(end-start)

except KeyboardInterrupt:
    print('interrupted!')



