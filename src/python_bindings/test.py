import cv2
import numpy as np
import np_opencv_module as npcv
from yolt_python import LaplacianBlending as fv
from matplotlib import pyplot as plt
from random import randint

center=[230, 150];
sigma_x=100
sigma_y=100
levels=4
img = cv2.imread('image.jpg')
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, channels = img.shape
img=plt.imread('image.jpg')
i=10;
try:
    while True:
    
        # RANDOM FIXATION POINTS
        center=[int(width/2.0), int(height/2.0)]
        # Convert np array to cv::Mat object
        my_mat_img = npcv.test_np_mat(img)
	# Create the Laplacian blending object
        my_lap_obj=fv(my_mat_img,levels,200,i*5)
        # Foveate the image
        foveated_img = my_lap_obj.foveate(npcv.test_np_mat(np.array(center)))
        # Display the foveated image
        plt.imshow(foveated_img)
        #img.set_data(im)

        circle=plt.Circle((center[0],center[1]),1.0,color='blue')
        ax = plt.gca()
        ax.add_artist(circle)


        plt.draw()
        plt.pause(.1)
        plt.cla()

        print('test')
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis


	i=i+1
        #plt.show()

except KeyboardInterrupt:
    print('interrupted!')



