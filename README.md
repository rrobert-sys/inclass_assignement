_**IN CLASS ASSIGNEMENT PROJECT FOR PR.MAJUMDAR AT USD**_

**The goal of this project was to analyze a set of at least 10 images using at least 2 Deep Learning Algorithm :**

I decided to use the YOLOv8-nano model and the Faster RCNN model, as they were for me the easiest to use and most beginner friendly.

The objectives was to see:

1) Their performance in terms of times(preprocess,inference, and postprocess),

2) The numbers of objects that can be detected,

3) Their probabilities,

4) Any additional features using deep learning or not: 
	In this case: Dominant colors (using Deep Learning) & Texture features (Edge Density, LBP, Sharpness).


**Below is the project structure:**

- **“pictures”**, which is the name of your file

- **"At home assignment.ipynb"**, which contains:

                                —yolo_detection (1st Model)

                                —Faster_RCNN (2nd Model)

                                —3 most dominants color from each detections 

                                —Textures features
-**README.md**

⚙️ **WHAT YOU WILL NEED**

⚠️(make sure that your pictures are in _**jpeg/jpg/png**_ formart)⚠️


🟦**INSTALLATIONS REQUIRED:**

pip install ultralytics

pip install torch torchvision

pip install matplotlib

pip install opencv-python

pip install scikit-learn

pip install scikit-image

pip install numpy

🟪 **_For Yolov8 nano model:_**

1.from ultralytics import YOLO

2.import matplotlib.pyplot as plt

3.import os

🟥 **_For the Faster RCNN:_**

1.import torch

2.from torchvision.models.detection import fasterrcnn_resnet50_fpn

3.import torchvision.transforms.functional as F

4.from PIL import Image

5.import matplotlib.pyplot as plt

6.import matplotlib.patches as patches

7.import time

8.import os


🟨 ** HOW TO MAKE THE PROJECT RUNNING**

- Place images inside the **pictures** folder

- Open the notebook: **At home assignment.ipynb**

- Run all the cells in order (very important)

- Each cell should be:
  
              - Yolo Model,
  
              - Faster RCNN Model,
  
              - 3 most dominant colors,
  
              - Textures Features


🏫 **OUTSIDE SOURCES**

**YOLO Explanation** (GeeksForGeeks):https://www.geeksforgeeks.org/computer-vision/how-does-yolo-work-for-object-detection/

**Ultralytics YOLO GitHub** : https://github.com/ultralytics/ultralytics

**Image Sharpening** : https://stackoverflow.com/questions/6646371/detect-which-image-is-sharper

**Local Binary Pattern** : https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/

**Faster RCNN explanation** : https://www.geeksforgeeks.org/machine-learning/faster-r-cnn-ml/

**Faster RCNN example**: https://github.com/trzy/FasterRCNN


<img width="640" height="360" alt="image" src="https://github.com/user-attachments/assets/41cbdd4d-4f0a-42d5-93ec-25a63d827cdb" />
