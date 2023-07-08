# YOLOv8 segmentation inference using Python

This is a web interface to [YOLOv8 object detection neural network](https://ultralytics.com/yolov8) 
implemented on [Python](https://www.python.org) via [ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-python.html).

This is a source code for a ["How to implement instance segmentation using YOLOv8 neural network"](https://dev.to/andreygermanov/how-to-implement-instance-segmentation-using-yolov8-neural-network-3if9) tutorial.

Watch demo: https://youtu.be/tv3mYPxj2n8

In addition, it includes all Jupyter Notebooks created in this article.

## Install

* Clone this repository: `git clone git@github.com:AndreyGermanov/yolov8_onnx_python.git`
* Go to the root of cloned repository
* Install dependencies by running `pip3 install -r requirements.txt`
* Download the [yolov8m-seg.onnx](https://drive.google.com/file/d/1uG1nagxQoyvcHfUYXcNDXJCvglsz7rdT/view?usp=sharing) model file to the app folder

## Run

Execute:

```
python3 object_detector.py
```

It will start a webserver on http://localhost:8080. Use any web browser to open the web interface.

Using the interface you can upload the image to the object detector and see bounding boxes and segmentation masks 
of all  objects detected on it.