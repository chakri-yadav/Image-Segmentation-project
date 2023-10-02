# Import necessary libraries
import os
import sys
import random
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import cv2
import time
from PIL import Image
from base64 import b64decode
from IPython.display import display, Javascript, Image

# Install the required library
!pip install mrcnn

# Import Mask R-CNN libraries
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config

# Set TensorFlow version to 1.x
%tensorflow_version 1.x

# Define the root directory
ROOT_DIR = os.path.abspath("")
sys.path.append(ROOT_DIR)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Configure inference settings
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create a model object in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Create a directory for the video frames
os.mkdir("video")

# COCO Class names
class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    # ...
    # Add all class names here
    # ...
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Open a video capture from a file (Specify your video file)
capture = cv2.VideoCapture("video/test-video-zoo.mp4")
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
fps = int(capture.get(cv2.CAP_PROP_FPS))  # Get frames per second
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video = cv2.VideoWriter("Zoo_video_masked.avi", fourcc, fps, (width, height))  # Use width and height from video

count = 0
while True:
    # Capture frame-by-frame
    ret, frame = capture.read()
    if not ret:
        break

    results = model.detect([frame], verbose=1)
    r = results[0]
    boxes = r['rois']
    masks = r['masks']
    class_ids = r['class_ids']
    scores = r['scores']

    start = time.time()
    masked_image = visualize.display_instances(frame, boxes, masks, class_ids, class_names, scores)
    end = time.time()
    print("Inference time: {:.2f}s".format(end - start))

    img = cv2.imread("Masked-image.jpg")
    print(img.shape)
    video.write(img)

    count += 1
    print("count:", count)

# Release the capture and video objects
print("WARNING")
cv2.destroyAllWindows()
capture.release()
video.release()

# Load another random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

# Function to capture a real-time image
def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = 'Capture';
            div.appendChild(capture);
            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
            await new Promise((resolve) => capture.onclick = resolve);
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getVideoTracks()[0].stop();
            div.remove();
            return canvas.toDataURL('image/jpeg', quality);
        }
        display(js)
        data = eval_js('takePhoto({})'.format(quality))
        binary = b64decode(data.split(',')[1])
        with open(filename, 'wb') as f:
            f.write(binary)
        return filename
    ''')

# Try to capture a real-time image
try:
    filename = take_photo()
    print('Saved to {}'.format(filename))
    image = skimage.io.imread(filename)

    results = model.detect([image], verbose=1)
    r = results[0]

    # Visualize results
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
except Exception as err:
    print(str(err))
