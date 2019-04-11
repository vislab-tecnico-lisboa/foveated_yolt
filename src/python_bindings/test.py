import cv2
import numpy as np
import np_opencv_module as npcv
from yolt_python import LaplacianBlending as fv
from matplotlib import pyplot as plt
from random import randint
import time

rho=-0.5
<<<<<<< HEAD
sigma_xx=50
sigma_yy=50
sigma_xy=int(np.floor(rho*sigma_xx*sigma_yy))

levels=7
img = cv2.imread('images/image_r_100.jpg')
=======
sigma_xx=80
sigma_yy=100
sigma_xy=int(np.floor(rho*sigma_xx*sigma_yy))

levels=7
img = cv2.imread('image.jpg')
>>>>>>> dd8896b40ba9d842d4f50bf1aac3e1a94d6302af
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, channels = img.shape

# Create the Laplacian blending object
my_lap_obj=fv(width,height,levels,sigma_xx,sigma_yy,sigma_xy)
try:
    while True:
        start = time.time()

        #sigma_x=randint(1, sigma_x_max)
        #sigma_y=randint(1, sigma_y_max)

        # RANDOM FIXATION POINTS
        center=[int(width/2.0), int(height/2.0)]
<<<<<<< HEAD
        #print npcv.test_np_mat(np.array(center))
=======
>>>>>>> dd8896b40ba9d842d4f50bf1aac3e1a94d6302af

        # RANDOM FOVEA SIZE
        #my_lap_obj.update_fovea(width,height,sigma_x,sigma_y)

        #print npcv.test_np_mat(np.array(center))
        # Foveate the image
        #print npcv.test_np_mat(np.array(center))
        foveated_img=my_lap_obj.foveate(img,npcv.test_np_mat(np.array(center)))

        end = time.time()
        #print(end - start)

        # Display the foveated image
        plt.imshow(foveated_img)
        #img.set_data(im)

        circle=plt.Circle((center[0],center[1]),1.0,color='blue')
        ax = plt.gca()
        #ax.add_artist(circle)

        plt.draw()
        plt.pause(.001)
        plt.cla()

except KeyboardInterrupt:
    print('interrupted!')



