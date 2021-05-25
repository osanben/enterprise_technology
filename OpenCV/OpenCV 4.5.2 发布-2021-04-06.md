**春季更新OpenCV 4.5.2发布了！**

来看看4.5.2都有哪些重要改进：

- **core模块**：增加并行后端的支持。特殊编译的OpenCV可以允许选择并行后端，并/或通过plugin动态载入。
- **imgpro模块**：增加智能剪刀功能（如下演示）。CVAT标注工具已经集成此功能，可在线体验https://cvat.org。

- **videoio模块**：改进硬件加速的视频编解码任务。从4.5.2开始，新的内置属性让用户更容易使用API

```
VideoCapture capture(filename, CAP_FFMPEG,
    {
        CAP_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_ANY,
    }
);
```

- **DNN模块：**

- - 改进TensorFlow解析错误的调试

  - 改进layers和activations，支持更多模型

  - - 优化NMS处理、DetectionOutput
    - 修复Div with constant、MatMul、Reshape(TensorFlow)
    - 支持Mish ONNX子图、NormalizeL2(ONNX)、LeakyReLU(TensorFlow)、TanH(Darknet)、SAM(Darknet)和Exp

  - 支持*OpenVINO* 2021.3 release支持

- **G-API模块：**

- - Python支持

  - - 引入新的Python后端：G-API可以运行Python的任意kernels作为pipeline的一部分
    - 扩展G-API Python绑定的推理支持
    - G-API Python绑定增加更多的图数据类型支持

  - 推理支持

  - - OpenVINO推理后端引入动态输入/CNN reshape功能
    - OpenVINO推理后端引入异步执行支持：推理可以在多个request并行运行以增加流密度/处理量
    - ONNX后端扩展对INT64/INT32数据类型的支持，OpenVINO后端扩展对INT32的支持
    - ONNX后端引入cv::GFrame/cv::MediaFrame和常量支持

  - 媒体支持

  - - 在绘制/渲染接口引入cv::GFrame/cv::Media支持
    - Streaming模式引入multi-stream输入支持以及帧同步以支持某些情况如Stereo
    - 增加Y和UV操作以访问图级别cv::GFrame的NV12数据；若媒体格式不同，转换是同时的

  - 运算符和核

  - - 增加新操作（MorphologyEx, BoundingRect, FitLine, FindLine, FindContours, KMeans, Kalman, BackgroundSubtractor）的性能测试
    - 修复PlaidML后端的RMat输入支持
    - 增加Fluid AbsDiffC, AddWeighted和位操作的ARM NEON优化

  - 其他静态分析和警告修复

- **文档：**

- - [GSoC]增加TF和PyTorch分类转换案例
  - [GSoC]增加TF和PyTorch分割转换案例
  - [GSoC]增加TF和PyTorch检测转换案例

- **社区贡献：**

- - core：增加带cuda stream标志的cuda::Stream构造函数
  - highgui：Win32上的OpenGL暴露VSYNC窗口属性
  - highgui：Win32上的pollKey()实现
  - imgcodecs：增加PNG的Exif解析
  - imgcodecs：OpenEXR压缩类型可选
  - imgproc：优化connectedComponents
  - videoio：Android NDK摄像头支持
  - (opencv_contrib)：腾讯微信QR码识别模块 
  - (opencv_contrib)：实现cv::cuda::inRange()
  - (opencv_contrib)：增加Edge Drawing Library中的算法
  - (opencv_contrib)：Viz模块增加Python绑定

更多详细信息请参考：

https://github.com/opencv/opencv/wiki/ChangeLog#version452