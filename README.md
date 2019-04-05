# Foveated-YOLT
You Only Look Twice - Foveated version

The pre-trained models used to test our method are CaffeNet, AlexNet, GoogLeNet and VGGNet (16 weight layers).



Donwload files and from root, create a build directory (mkdir build).
Execute from root

bash scripts/setup.sh to directly download the pre-trained models.

To compile from root: 

```
cd build
cmake ..
make
```

To run yolt.cpp from root:

```
bash scripts/setup.sh
```

```
bash scripts/run_detector.sh
```

To configure your network and its parameters, change the ```run_detector.sh``` file accordingly.

If you use our code, please cite our work:

```
@inproceedings{almeida2017deep,
  title={Deep Networks for Human Visual Attention: A hybrid model using foveal vision},
  author={Almeida, Ana Filipa and Figueiredo, Rui and Bernardino, Alexandre and Santos-Victor, Jos{\'e}},
  booktitle={Iberian Robotics conference},
  pages={117--128},
  year={2017},
  organization={Springer}
}
```



