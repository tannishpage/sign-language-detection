# Setup

Before running the demo, modify the following lines (3 to 8) in demo.py
```
ROOT_DIR = "path to mask rcnn tf 2.0 repo"
PATH_TO_RNN_MODEL = "./rnn_75_frame_chunck.h5"
vid = "PATH TO TEST VIDEO"
sys.path.append("path to Sign_Language_Detection folder") # This repo
sys.path.append(ROOT_DIR)
sys.path.append("path to SORT repo")
```
# Dependencies
1. [Mask-RCNN Tensorflow 2.0](https://github.com/leekunhee/Mask_RCNN)
2. [SORT](https://github.com/abewley/sort)
3. Tensorflow 2.5.0
4. numpy
5. scikit-image (skimage)
6. matplotlib
7. opencv

Dependencies 3 to 7 can be installed using `pip install tensorflow=2.5.0 opencv-python numpy scikit-image matplotlib`
