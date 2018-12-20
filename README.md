# HyperFace
A TensorFlow implementation of the paper:  HyperFace: A Deep Multi-task Learning Framework for Face Detection, Landmark Localization, Pose Estimation, and Gender Recognition

#### Dependencies:
* Tensorflow
* dlib
* scikit-learn
* OpenCV

How to install `dlib` package:
```
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev
pip install dlib
```

#### Data:
Download the sample data and save it at `data/hyf_data.npy`

#### Execution:
1. `cd src`
2. `python train.py`