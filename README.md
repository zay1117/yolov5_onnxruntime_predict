# onnxruntime推理yolov5
## 1.所需环境
g++ ：11.4.0；
cmake：3.22.1；
CUDA：11.8；
OpenCV：4.9.0；
onnxruntime：1.12.0 (这个最好是这个版本，因为onnxruntime 1.13.0以上的版本有些函数更改，需要修改部分代码)；
## 2. 代码运行（比较适合初学者）
1.模型文件位于models文件夹中，有yolov5s.onnx和yolov5n.onnx两种；
2.模型预测主要分为摄像头预测和单张图像预测，为了简便，需要在main.cpp文件夹中修改参数useCamera，当其为true以打开摄像头模型，false为处理单张图像；
3.参数主要在main.cpp中进行修改，例如置信度和iou值；
## 1. Required Environment
g++: 11.4.0
CMake: 3.22.1
CUDA: 11.8
OpenCV: 4.9.0
ONNX Runtime: 1.12.0 (It is recommended to use this version because versions 1.13.0 and later have changes in some functions that may require code modifications.)
## 2. Running the Code (Beginner-Friendly)
1.Model Files: The model files are located in the models folder, including yolov5s.onnx and yolov5n.onnx.
2.Prediction Modes: The code supports two modes of prediction: camera-based and single-image. To switch between these modes, modify the useCamera parameter in the main.cpp file:
   ——Set useCamera to true to enable camera-based prediction.
   ——Set useCamera to false for single-image prediction.
3.Parameters: You can adjust various parameters, such as confidence and IoU thresholds, in the main.cpp file.
