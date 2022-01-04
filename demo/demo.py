# The demo script for AUSLAN Detection and identification
import sys
ROOT_DIR = "path to mask rcnn tf 2.0 repo"
PATH_TO_RNN_MODEL = "./rnn_75_frame_chunck.h5"
vid = "PATH TO TEST VIDEO"
sys.path.append("path to Sign_Language_Detection folder")
sys.path.append(ROOT_DIR)
sys.path.append("path to SORT repo")

from create_neural_net_model import create_cnn_model, create_neural_net_model

from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgba2rgb
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
#sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
#import coco

# Tracking algorithm
import sort

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

tracker = sort.Sort() #Using default tracker settings

# In[2]:


print(MODEL_DIR)
print(tf.__version__)


# ## Configurations
#
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
#
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[3]:


class InferenceConfig(config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = 'coco'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

conf = InferenceConfig()
conf.display()


# ## Create Model and Load Trained Weights

# In[4]:

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=conf)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

cnn_model = create_cnn_model((224, 224, 3), True) # Feature Extractor
# Sign Language Detection Model

rnn_model = create_neural_net_model((224, 224, 3),
                                    (75, 224, 224, 3),
                                    (75, 1000),
                                    include_convolutional_base=False,
                                    rnn_model_weights_file=PATH_TO_RNN_MODEL)


import cv2
import time
cap = cv2.VideoCapture(vid)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./output.avi',fourcc, 25.0, (1920,1080))
color = [(0, 256, 0)]
count = 1
frames = []
cnn_preds = {}
print("Collecting frames")
while True:
    ret, frame = cap.read()
    if ret:
        count += 1
        print(count)
        results = model.detect([frame])
        frames.append(frame)
        r = results[0]
        indices = np.where(r['class_ids'] == 1) # Getting human class only
        bounding_boxes = []
        for i, index in enumerate(indices[0]):
            y1, x1, y2, x2 = r['rois'][index]
            #print("({}, {}, {}, {}, {})".format(x1, y1, x2, y2, r['scores'][index]))
            bbox = (x1, y1, x2, y2, r['scores'][index])
            bounding_boxes.append(bbox)
        output = tracker.update(np.array(bounding_boxes))
        for human in output:
            # Save the humans to the frames dict.
            x1, y1, x2, y2 = int(human[0]), int(human[1]), int(human[2]), int(human[3])
            X = preprocess_input(cv2.resize(frame[y1:y2, x1:x2], (224, 224)))
            X = np.expand_dims(X, axis=0)
            if human[4] in cnn_preds.keys():
                cnn_preds[human[4]][0].append(cnn_model.predict_on_batch(X)[0])
                cnn_preds[human[4]][1].append((x1, y1, x2, y2))
            else:
                cnn_preds[human[4]] = ([cnn_model.predict_on_batch(X)[0]], [(x1, y1, x2, y2)])
        if len(frames) == 75:
            rnn_results = []
            print("RNN_predicting")
            for key in cnn_preds:
                if len(cnn_preds[key]) != 75:
                    continue # Skip, it's not big enough to make a prediction
                cnn_preds_array = np.array(cnn_preds[key][0])
                cnn_preds_array = np.expand_dims(cnn_preds_array, axis=0)
                rnn_result = rnn_model.predict(cnn_preds_array)
                if np.argmax(rnn_result[0]) == 1:
                    rnn_results.append(key)
            print("Drawing Bounding Boxes")
            for i, f in enumerate(frames):
                for key in rnn_results:
                    x1, y1, x2, y2 = cnn_preds[key][1][i]
                    cv2.rectangle(f, (x1, y1), (x2, y2), color[0], 2)
                out.write(f)
                cv2.imshow('window', f)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            print("Processed", count, "frames")
            print("Collecting frames")
            frames = []
            cnn_preds = {}
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
