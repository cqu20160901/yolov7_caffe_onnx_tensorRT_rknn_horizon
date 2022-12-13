# yolov7_caffe_onnx_tensorRT_rknn_Horizon

yolov7 部署版本，后处理用python语言和C++语言形式进行改写，便于移植不同平台（caffe、onnx、tensorRT、rknn、Horizon）。

# 文件夹结构说明

yolov7_caffe：去除维度变换层的prototxt、caffeModel、测试图像、测试结果、测试demo脚本

yolov7_onnx：onnx模型、测试图像、测试结果、测试demo脚本

yolov7_TensorRT：TensorRT版本模型、测试图像、测试结果、测试demo脚本、onnx模型、onnx2tensorRT脚本(tensorRT-7.2.3.4)

yolov7_rknn：rknn模型、测试（量化）图像、测试结果、onnx2rknn转换测试脚本

yolov7_horizon：地平线模型、测试（量化）图像、测试结果、转换测试脚本、测试量化后onnx模型脚本

# 测试结果
![image](https://github.com/cqu20160901/yolov7_caffe_onnx_tensorRT/blob/main/yolov7_caffe/result.jpg)

说明：预处理没有考虑等比率缩放和以及BGR2RGB
