# HyperFace
A TensorFlow implementation of the following paper by Rajeev Ranjan, Vishal M. Patel and Rama Chellappa published in _TPAMI_:  
[HyperFace: A Deep Multi-task Learning Framework for Face Detection, Landmark Localization, Pose Estimation, and Gender Recognition](https://arxiv.org/abs/1603.01249)

#### Dependencies:
* Tensorflow
* dlib
* scikit-learn
* OpenCV

Steps to install dependencies:
```
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev
pip install -r requirements.txt
```

#### Data:
Download the sample data and save it at `data/hyf_data.npy`

#### Execution:
1. `cd src`
2. `python train.py`