# yolov7_caffe_onnx_tensorRT
yolov7 部署版本，后处理用python语言和C++语言形式进行改写，便于移植不同平台（caffe、onnx、tensorRT）。

# 文件夹结构说明

yolov7_caffe：去除维度变换层的prototxt、caffeModel、测试图像、测试结果、测试demo脚本

yolov7_onnx：onnx模型、测试图像、测试结果、测试demo脚本

yolov7_TensorRT：TensorRT版本模型、测试图像、测试结果、测试demo脚本、onnx模型、onnx2tensorRT脚本(tensorRT-7.2.3.4)

# 测试结果
![image](https://github.com/cqu20160901/yolov7_caffe_onnx_tensorRT/blob/main/yolov7_caffe/result.jpg)

说明：预处理没有考虑等比率缩放和以及BGR2RGB
