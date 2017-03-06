# Foveated-YOLT
You Only Look Twice - Foveated version

The pre-trained models used to test our method are CaffeNet, AlexNet, GoogLeNet and VGGNet (16 weight layers).



Donwload files and from root, create a build directory (mkdir build).

Execute from root


bash scripts/setup.sh to directly download the pre-trained models.



To compile from root: 

cd build

cmake ..

make




To run thesis.cpp from root:

bash scripts/run_detector.sh
 


To configurate your network and its parameters, change accordingly the run_detector.sh file.



