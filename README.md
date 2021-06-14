# 机器人真·涨姿势了：比肩人类抓取能力，上海交大、非夕科技联合提出全新方法AnyGrasp

在近日召开的 ICRA （国际机器人与自动化会议）大会上，上海交大-非夕科技联合实验室展示了最新研究成果「AnyGrasp」（[https://graspnet.net/anygrasp.html](https://graspnet.net/anygrasp.html)），第一次实现机器人对于任意场景的任意物体的通用高速抓取，在机械臂硬件构型、相机不作限制的情况下，让机器人拥有比肩人类抓取能力的可能。

基于视觉的机器人通用抓取，一直是学界和业界的关注重点，也是机器人智能领域亟待解决的问题之一。

针对物体抓取，业界通常需要先对物体进行三维建模，然后训练网络，在实际中先进行位姿检测，再进行抓取：

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210614150310.png)

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210614150332.png)

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210614150409.png)

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210614150423.png)

此前对于简单场景简单物体的抓取，学术界也有研究涉猎。但是机器人日常面对的都是大量堆叠的、复杂的、没有见过的物体，同时场景呈现极度的杂乱性，还没有相关研究可直接面对任意复杂场景进行抓取。

我们能否期待一个通用算法，能像人类一样具备面向任意场景、任意物体的抓取能力？ 

譬如，当杯子被敲碎，每个碎片都是未曾出现过的，机器人可以将这些从未见过、更未被建模的碎片一片片捡起来：

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210614150737.png)

*搭载AnyGrasp的机器人首秀*

同时，它还要能适应更多的不确定性。比如一堆来自新疆戈壁滩的玛瑙石，细小且局部复杂：

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210614150759.png)

再比如在日常场景经常会遇到的会随机形变的柔性袋装零食或者布娃娃：

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210614150824.png)

以及各种玩具、五金件、日常用品：

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210614150842.png)

甚至人造的形状复杂的对抗样本 [1]：

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210614150901.png)

更极端的，如果光照情况不好，同时有探照灯的干扰，桌面还会变化，机器人能不能稳定地抓取？

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210614150919.png)

在这些方面，尚且没有研究能达到稳定的抓取效果，甚至没有前期可论证的 DEMO。此前来自 UCB 的研究团队发表于《Science Robotics》的成果 DexNet4.0 [2]，也只局限于低自由度的垂直抓取，同时需要搭配价值数万元的工业级高精度深度相机，计算一次耗时数十秒。

近日，上海交大-非夕科技联合实验室在 ICRA 大会上展示了最新研究成果「AnyGrasp」，基于二指夹爪的通用物体抓取。这是**第一次机器人对于任意场景的任意物体，有了比肩人类抓取的能力，**无需物体 CAD 模型与检测的过程，对硬件构型、相机也没有限制。

仅需要一台 1500 元的 RealSense 深度相机，AnyGrasp 即可在数十毫秒的时间内，得到其观测视野内整个场景的数千个抓取姿态，且均为六自由度，以及一个额外的宽度预测。在五小时复杂堆叠场景的抓取中，单臂 MPPH（Mean Pick Per Hour, 单位小时内平均抓取次数）可达到 850+，为 DexNet4.0 的三倍多，这是该指标第一次在复杂场景抓取上接近人类水平（900-1200 MPPH）。

以下为搭载 AnyGrasp 的最新成果展示，在六轴机械臂上：

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210614151004.png)

在七轴机械臂上：

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210614151042.png)

在ICRA2021的展区内，搭载AnyGrasp的机器人更是走出了实验室，在现场直接对没见过的物体进行抓取，同时与参会观众进行互动，由现场观众自由发挥，用随身的物品、捏的橡皮泥对它进行考验，机器人都能进行稳定的抓取。

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210614151120.png)

研究团队介绍，目前 AnyGrasp 有以下优势：

- 普适性：对未曾见过、复杂场景中的任意物体均可抓取，包括刚体、可变形物体、无纹理的物体等；
- 高速度：数十毫秒内即可生成数千个稳定的抓取姿态；
- 稳定性：对背景、光照、桌面角度等不敏感；
- 低成本：无需高精度工业相机，千元价位的深度相机（如 Intel RealSense）即可胜任。

技术层面上，AnyGrasp 的实现是基于研究团队提出的一个全新方法论，即真实感知与几何分析的孪生联结。真实感知与密集几何标注原本是矛盾的两方面，因为真实感知往往需要人工标注，而几何分析需依赖仿真环境，此前未曾有团队在这方面进行过尝试。

在 CVPR 2020 会议上，上海交大团队提出了 GraspNet-1Billion 数据集 [3]，其中包含数万张单目摄像头采集的真实场景的 RGBD 图像，每张图片中包含由基于物理受力分析得到的数十万个抓取点，数据集中总共包含超过十亿有效抓取姿态。为了达到真实感知与几何分析的孪生联结目标，团队设计了一个半自动化的数据收集与标注方法，使得大规模地生成包含真实视觉感知与物理分析标签的数据成为可能。该数据集及相关代码目前已经开源。

基于 GraspNet-1Billion 数据集，团队开发了一套新的可抓取性（graspness）嵌入端到端三维神经网络结构，在单目点云上直接预测整个场景可行的抓取姿态，根据采样密度，抓取姿态可从数千到数万不等，整个过程仅需数十毫秒。基于全场景的密集的抓取姿态，后续任务可根据目标及运动约束选择合适的抓取位姿。

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210614151146.png)

目前，AnyGrasp 还在持续开发与迭代中，后续研究团队计划构建抓取算法数据社区 ，并开放抓取任务在线测评。相关的学术数据集、SDK、学术算法库将在 **www.graspnet.net** 上开放。

参考链接：

【1】EGAD! an Evolved Grasping Analysis Dataset for diversity and reproducibility in robotic manipulation，Douglas Morrison , Peter Corke , Jurgen Leitner,IEEE Robotics & Automation Letters, 2020

【2】Learning ambidextrous robot grasping policies, Jeffrey Mahler, Matthew Matl, Vishal Satish, Michael Danielczuk, Bill DeRose, Stephen McKinley, Ken Goldberg, Science Robotics, 2019

【3】GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping, Hao-Shu Fang; Chenxi Wang; Minghao Gou; Cewu Lu, CVPR, 2020

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210608112105.png)