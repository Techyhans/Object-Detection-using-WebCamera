# Realtime Detection with Browser
This is a web based application where you able to perform real-time object detection usinmg camera.

## Requirements
- YOLOv4 Pretrained Model -- Download [here](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT).
- Flask == 1.1.2
- Flask-Cors == 3.0.9
- Opencv-Python == 4.4.0
- Numpy == 1.19.4

## Pretrained Model
Weights file needed to place at the same directory with the source code.
YOLOv4 pretrained model can download [here.](https://github.com/AlexeyAB/darknethttps://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) 

## How to run?
1. Clone this repository.
2. Environment setup and download YOLOv4 model.
3. Place YOLOv4 model to the same directory of api.py.
4. Run command `python api.py`.
5. Open browser and insert `http://localhost:5000`.
6. Give access to camera and see the result!

## Develop By
TechyHans - https://techyhans.com
