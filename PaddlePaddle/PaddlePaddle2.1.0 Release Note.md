## 重要更新

飞桨框架2.1.0 版本有如下重要更新：

- 环境适配： 增加了对Python 3.9、CUDA 11.2的支持；提供了对[ROCm平台](https://rocmdocs.amd.com/en/latest/)的支持（experimental）；提供了对[昇腾AI处理器](https://e.huawei.com/cn/products/cloud-computing-dc/atlas/ascend-910)的支持（experimental）；增加了可在[百度昆仑芯片](https://cloud.baidu.com/product/kunlun.html)上运行的模型数量；详情请见：[开始使用](https://www.paddlepaddle.org.cn/install/quick)。

- 分布式训练：在已有静态图的[多维混合并行](https://mp.weixin.qq.com/s/BblzcVn0NQ-QIhywvmoOuA)的基础上，新增动态图实现。

- 框架功能：完成了多项功能增强和性能优化，特别的，新增了以下重要功能：
  - 自定义算子：提供了在框架外部自定义算子的新方案，简化了自定义算子写法与训练推理部署流程，详情请见：[自定义外部算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/07_new_op/new_custom_op_cn.html)。
  - 新增inplace操作：新增可降低显存占用与提升性能的inplace操作，包括View策略，与12个inplace API。
  - 高层API相关：新增支持混合精度训练的高层API；新增通过`paddle.hub`来查看、共享、加载模型。
  - 自动混合精度训练优化： 优化了混合精度训练中slice、where、range等多个op的计算性能，提升了在MaskRCNN、ERNIE等模型上的加速效果。
  - oneDNN下BF16训练：新增支持了AMP(AutoMixedPrecision) pure_BF16模式; 新增支持了BF16类型的SGD和initializers初始值设定并减小了内存；新增支持了大部分word2vec BF16训练需要的前向和反向op。

飞桨的官方模型库和套件的最新更新请参见：[Paddle projects notes along with PaddlePaddle2.1](https://github.com/PaddlePaddle/Paddle/wiki/Paddle-projects-notes-along-with-PaddlePaddle2.1)。

## 不兼容升级

- 飞桨框架2.1放弃了对python2和python3.5的支持，建议您升级python到3.8版本来使用飞桨。飞桨框架2.1不再提供支持CUDA9的预编译包，建议您升级CUDA版本来使用飞桨。
- 对API可见性的优化，会导致无法使用`from deeply_nested_namespace import *`的方式导入被认为是实现细节的位于最底层的命名空间中的私有API。建议您通过查看飞桨官网的[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)说明来使用飞桨。具体的，以下行为在飞桨框架2.1版本中不再被允许。

## 训练框架

### 功能优化（含分布式）

#### 基础API

- 新增`paddle.dtype` 以及 `paddle.float32` 等数据类型，作为 paddle 内的数据类型。 ([#32012](https://github.com/PaddlePaddle/Paddle/pull/32012))
- 新增`paddle.nn.functional.glu`。 ([#32096](https://github.com/PaddlePaddle/Paddle/pull/32096))
- 新增`paddle.nn.utils.spectral_norm`。[#32633](https://github.com/PaddlePaddle/Paddle/pull/32633)
- 新增`paddle.Tensor.register_hook` API，用于在动态图场景中为前向Tensor对应的梯度Tensor注册hook函数。([#31775](https://github.com/PaddlePaddle/Paddle/pull/31775))
- 新增`Tensor.__array__`函数，支持`numpy.array(Tensor)`和`numpy.asarray(Tensor)`将`paddle.Tensor`类型转换成`numpy.ndarray`类型 。([#32300](https://github.com/PaddlePaddle/Paddle/pull/32300))

#### 高层API

- 新增`paddle.hub`功能，提供`help`、`list`和`load`函数用于查看和加载第三方模型，支持加载远程和本地repository。([#31873](https://github.com/PaddlePaddle/Paddle/pull/31873))
- 支持混合精度训练，提供O0, O1, O2三种模式，分别对应FP32训练、自动混合精度训练、纯FP16训练。目前纯FP16训练仅支持静态图。([#31417](https://github.com/PaddlePaddle/Paddle/pull/31417))
- 支持`paddle.Tensor`类型的图像变换，包括`normalize, to_grayscale, vflip, hflip, crop, center_crop, pad, rotate, resize`等算子 。([#32705](https://github.com/PaddlePaddle/Paddle/pull/32705))

#### 动态图转静态图

修复了动态图转静态图的bug：

- 静态图`arange、range` API返回的shape与动态图不一致。
- `paddle.to_tensor`在动转静中支持输入为`int，float，bool`基础类型。
- for循环中支持解析dict推导式语法。([#32159](https://github.com/PaddlePaddle/Paddle/pull/32159))
- 修复部分场景下嵌套控制流语句中存在变量未声明报错的问题。([#32153](https://github.com/PaddlePaddle/Paddle/pull/32153))
- 修复了`expand` op缺少float16类型的bug。([#32238](https://github.com/PaddlePaddle/Paddle/pull/32238))
- 修复了`expand_v2、tile、expand、expand_as、expand_as_v2、meshgrid`等6个OP反向梯度求解，当shape维度为6时，返回梯度信息为None的bug。([#32004](https://github.com/PaddlePaddle/Paddle/pull/32004))
- 修复了`paddle.jit.TraceLayer.save_inference_model`接口中因未同时保存网络结构和参数导致与`paddle.static.load_inference_model`搭配使用不一致的问题。([#31989](https://github.com/PaddlePaddle/Paddle/pull/31989) )

#### 混合精度训练

- 动态图混合精度接口 auto_cast 中自动将不支持fp16 kernel的op保持为fp32计算。([#32543](https://github.com/PaddlePaddle/Paddle/pull/32543))
- 修复静态图混合精度训练中因不支持FP16计算的Op列表(`unsupported_fp16_list`)统计不完整导致的意外报错问题，当前不支持FP16计算的Op列表可根据运行时环境自动生成。([#32102](https://github.com/PaddlePaddle/Paddle/pull/32102))
- 优化`update_loss_scaling` for循环起多个相同cuda kernel问题，融合为一个cuda kernel。([#32554](https://github.com/PaddlePaddle/Paddle/pull/32554))
- 优化`slice`多维情况下性能较慢问题。([#32266](https://github.com/PaddlePaddle/Paddle/pull/32266))
- 优化`elementwise_add_grad`输入输出相同时的冗余拷贝问题。([#32051](https://github.com/PaddlePaddle/Paddle/pull/32051))
- 优化`check_finite_and_unscale` for循环起多个相同cuda kernel问题，融合为一个cuda kernel。([#31954](https://github.com/PaddlePaddle/Paddle/pull/31954))
- 优化`range`参数冗余拷贝问题。([#30811](https://github.com/PaddlePaddle/Paddle/pull/30811))
- 优化`top_k_v2`在`input_width <= 1024`时性能较慢问题。([#30403](https://github.com/PaddlePaddle/Paddle/pull/30403))
- 移植`where_index` CPU计算流程到GPU上完成。([#30601](https://github.com/PaddlePaddle/Paddle/pull/30601))

#### BF16训练

- 增加了初级 BF16 AMP 集成, 通过在前向网络中添加`cast op`来修改图使一些 operator 使用 BF16 kernel 。([#31093](https://github.com/PaddlePaddle/Paddle/pull/31093))
- 增加了 BF16 `pure_mode`模式, 在此模式下，默认开启使用 BF16 数据类型的模型参数，BF16的operator，对于optimizer的BF16 decorator。([#32281](https://github.com/PaddlePaddle/Paddle/pull/32281), [#32681](https://github.com/PaddlePaddle/Paddle/pull/32681))
- 增加了对于CPU flags的检查以确认是否支持oneDNN BF16性能提升。([#30551](https://github.com/PaddlePaddle/Paddle/pull/30551))
- 对BF16支持进行过程统一。([#31034](https://github.com/PaddlePaddle/Paddle/pull/31034))
- 增加了对于constant initilizer的BF16数据类型的支持。([#31935](https://github.com/PaddlePaddle/Paddle/pull/31935))

#### 分布式训练优化

- 加入图检索引擎，支持万亿边规模的分布式图神经网络存储、采样、训练([#31226](https://github.com/PaddlePaddle/Paddle/pull/31226))。
- 加入基于索引的数据采样类，支持图、树深度匹配等模型的采样([#31696](https://github.com/PaddlePaddle/Paddle/pull/31696))。
- 新增`paddle.distributed.send, paddle.distributed.recv，paddle.distributed.new_group，paddle.distributed.wait`，完善分布式通信API。([#32504](https://github.com/PaddlePaddle/Paddle/pull/32504), [#31682](https://github.com/PaddlePaddle/Paddle/pull/31682))

#### 自定义OP

- 新增支持Mac平台上使用自定义OP功能。([#31976](https://github.com/PaddlePaddle/Paddle/pull/31976))。
- Mac平台下支持C++/v11头文件目录的自动搜索功能，兼容本地可能存在多版本clang的情况。
- 新增支持Op前反向函数Attribute参数以及inferShape, InferDtype函数输入参数使用const &类型。([#31588](https://github.com/PaddlePaddle/Paddle/pull/31588))
- 新增支持在自定义Op实现时使用三种框架内部数据类型`paddle::complex64, paddle::complex128, paddle::float16`。([#31602](https://github.com/PaddlePaddle/Paddle/pull/31602), [#31657](https://github.com/PaddlePaddle/Paddle/pull/31657), [#31669](https://github.com/PaddlePaddle/Paddle/pull/31669), [#31725](https://github.com/PaddlePaddle/Paddle/pull/31725))

#### 模型保存与载入

- `paddle.save, paddle.load`支持Tensor的保存加载。([#31756](https://github.com/PaddlePaddle/Paddle/pull/31756))
- `paddle.save, paddle.load`支持`list[Tensor]、dict[Tensor]、tuple[Tensor]`以及`list、tuple、dict`嵌套的包含Tensor的结构的保存加载。([#32446](https://github.com/PaddlePaddle/Paddle/pull/32446))
- `paddle.save, paddle.load`支持Layer的保存加载。([#32446](https://github.com/PaddlePaddle/Paddle/pull/32446))

### 性能优化（含分布式）

- 优化重点算子，提升多个模型单GPU训练性能，Deeplabv3+单卡FP32和AMP性能分别提升11%、72%，TSM单卡AMP性能提升44.5%，HRNet单卡FP32、AMP分别提升46%、51%。
- 增加 `index_sample` CUDA实现。([#30380](https://github.com/PaddlePaddle/Paddle/pull/30380))
- 实现`relu, leaky_relu`算子的CUDA Kernel，代替原Eigen实现，正反向共提升5% ～ 20%。([#31869](https://github.com/PaddlePaddle/Paddle/pull/31869), [#31841](https://github.com/PaddlePaddle/Paddle/pull/31841))
- `temporal_shift` 性能提升20%～40%。([#31642](https://github.com/PaddlePaddle/Paddle/pull/31642))

## 推理部署

### 模型量化

- 新增支持将FP32模型保存为FP16模型。([#32112](https://github.com/PaddlePaddle/Paddle/pull/32112))
- 重构动态图量化训练中统计输出量化信息模块，支持多Block和多分支的模型，增强通用性。([#31680](https://github.com/PaddlePaddle/Paddle/pull/31680) [#31710](https://github.com/PaddlePaddle/Paddle/pull/31710) [#31784](https://github.com/PaddlePaddle/Paddle/pull/31784) [#31861](https://github.com/PaddlePaddle/Paddle/pull/31861))
- 动态图量化训练功能支持跳过量化OP，并且和预测端形成打通。([#31704](https://github.com/PaddlePaddle/Paddle/pull/31704))

### Paddle Inference

#### 功能升级

- 发布C API (experimental)， 功能与C++ API基本对齐。([#32225](https://github.com/PaddlePaddle/Paddle/pull/32225))
- 重构Tensor 底层代码，与旧有 ZeroCopyTensor 数据结构解耦。此升级不涉及用户 API 改动，对用户透明。([#31402](https://github.com/PaddlePaddle/Paddle/pull/31402))
- 预测框架python接口接入训练自定义算子。用户在训练过程中加载自定义算子后，即可像框架原生算子那样，通过 PaddlePredictor 直接执行包含此自定义算子的预测模型部署。([#32533](https://github.com/PaddlePaddle/Paddle/pull/32533))
- 支持从内存加载模型时TensorRT序列化和反序列化功能。([#31342](https://github.com/PaddlePaddle/Paddle/pull/31342))

#### 性能优化

- 支持ERNIE量化模型在NV GPU上混合精度推理，其中MatMul以Int8精度计算，其他部分以FP16精度计算。相比纯FP16推理，在T4上batch size=40时，标准ERNIE模型在XNLI数据集上推理性能由1898 seq/s提升至2310 seq/s，提升17.8%。([#32232](https://github.com/PaddlePaddle/Paddle/pull/32232))

#### 易用性优化

- 用户开启TensorRT变长输入，输入shape超出限定范围时增加报错信息。([#32155](https://github.com/PaddlePaddle/Paddle/pull/32155))
- 增加运行时TensorRT版本检查，若运行和编译时TensorRT大版本号不一致会以warning提示。([#32443](https://github.com/PaddlePaddle/Paddle/pull/32443))
- 增加TensorRT VERBOSE级别log开关，用户可通过`export GLOG_v=3`开启TensorRT VERBOSE日志，打印更多调试信息。([#32459](https://github.com/PaddlePaddle/Paddle/pull/32459))

## 环境适配

### 编译安装

- 新增支持CUDA11.2编译，支持3070/3080/3090显卡架构的编译。([#31529](https://github.com/PaddlePaddle/Paddle/pull/31529))
- 新增支持Windows Visual Studio 2017编译，并将发版、CI/CE、编译文档等各项配套设施，由VS2015全面升级至VS2017。([#311652](https://github.com/PaddlePaddle/Paddle/pull/31652))
- 新增对cuda11.2镜像的支持。([#32531](https://github.com/PaddlePaddle/Paddle/pull/32531))
- cuda10.1镜像支持gcc 5.4。([#32531](https://github.com/PaddlePaddle/Paddle/pull/32531))
- 镜像中新增对python 3.9的支持。([#32385](https://github.com/PaddlePaddle/Paddle/pull/32385))
- 修复`run_check`接口的bug，并在`run_check`接口里新增了对动态图的检查：现在`run_check`检测paddle安装的逻辑里，首先检测用户机器上是否有GPU，没有则报warning，未考虑安装cpu包的用户。([#32428](https://github.com/PaddlePaddle/Paddle/pull/32428))
- 修复Windows系统上缺乏 symlink 方法的问题。([#31006](https://github.com/PaddlePaddle/Paddle/pull/31006))

### 新硬件训练支持

- 新增支持海光芯片：飞桨基于 ROCM 4.0.1 版本可以在海光CPU与DCU上进行模型训练与推理。已经验证支持图像分类、目标检测、图像分割、自然语言处理、推荐系统、视频分类与语音合成共计7个分类的36个模型。([#29342](https://github.com/PaddlePaddle/Paddle/pull/29342), [#30758](https://github.com/PaddlePaddle/Paddle/pull/30758), [#30639](https://github.com/PaddlePaddle/Paddle/pull/30639), [#31009](https://github.com/PaddlePaddle/Paddle/pull/31009), [#31077](https://github.com/PaddlePaddle/Paddle/pull/31077))
- 新增支持昇腾芯片：支持在昇腾NPU上进行单机多卡训练。([#31957](https://github.com/PaddlePaddle/Paddle/pull/31957), [#32381](https://github.com/PaddlePaddle/Paddle/pull/32381), [#32197](https://github.com/PaddlePaddle/Paddle/pull/32197), ...)
- 昆仑硬件训练支持
  - 昆仑XPU支持动态图分布式训练。([#30455](https://github.com/PaddlePaddle/Paddle/pull/30455), [#30671](https://github.com/PaddlePaddle/Paddle/pull/30671))
  - 昆仑XPU支持fleet分布式训练。([#30858](https://github.com/PaddlePaddle/Paddle/pull/30858))
  - 昆仑XPU支持spawn启动多卡训练，优化XPU动态图多卡性能。([#31130](https://github.com/PaddlePaddle/Paddle/pull/31130))
  - 昆仑XPU静态图多卡支持fuse allreduce及gradient merge优化。([#31104](https://github.com/PaddlePaddle/Paddle/pull/31104))
  - 支持昆仑XPU暴露all_reduce/reduce集合通信API。([#32303](https://github.com/PaddlePaddle/Paddle/pull/32302))
  - 修复昆仑XPU动态图多卡随机hang住的bug。([#32662](https://github.com/PaddlePaddle/Paddle/pull/32662))