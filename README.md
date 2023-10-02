# Mask R-CNN Image and Video Object Detection

This repository contains Python code for object detection in images and videos using Mask R-CNN with pre-trained weights. Mask R-CNN is a deep learning model for object instance segmentation and can detect objects in images and videos while providing pixel-level masks.

## Requirements

Before running the code, make sure you have the following dependencies installed:

- Python 3.x
- TensorFlow 1.x (You can switch TensorFlow versions using `%tensorflow_version 1.x`)
- OpenCV (cv2)
- NumPy
- scikit-image
- matplotlib
- Pillow (PIL)
- IPython

You can install most of these dependencies using `pip`. For TensorFlow 1.x, you may need to use a suitable environment with TensorFlow 1.x installed.

```bash
pip install opencv-python-headless numpy scikit-image matplotlib pillow ipython
```

Additionally, you will need to install the Mask R-CNN library. You can do this with:

```bash
pip install mrcnn
```

## Getting Started

1. Clone this repository or download the code.

2. Download the pre-trained COCO weights for Mask R-CNN from the official website and save them as `mask_rcnn_coco.h5` in the project root directory. If the file does not exist, the code will download it for you when you run it for the first time.

3. Organize your image and video files. Images for object detection should be placed in the `images` directory, and videos should be placed in the `video` directory.

4. Run the code by executing the provided Python script.

```bash
python mask_rcnn_image_and_video.py
```

## Usage

- The code can perform object detection on both images and videos. Detected objects will be highlighted with bounding boxes and pixel-level masks.

- You can configure various parameters such as the input video file, output video file format, and object classes to detect by modifying the code.

- The code also includes a function to capture a real-time image using your webcam. This functionality requires an IPython environment.

- To capture a real-time image, run the code, and follow the instructions to click the "Capture" button when prompted. The captured image will be displayed and saved as `photo.jpg`.

- You can customize the code to suit your specific object detection requirements.

## Acknowledgments

This code is based on the Mask R-CNN implementation provided by Matterport, Inc. More information about Mask R-CNN can be found at [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN).

## License

This code is provided under the MIT License. You are free to use, modify, and distribute the code as per the terms of the license.

Please make sure to respect the licensing terms of the libraries and datasets used in this project, especially if you use it for commercial purposes.

Enjoy object detection with Mask R-CNN!
