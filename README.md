# 7 Papers & Radios | 史上最强三维人脑地图；3D渲染图转真实图像

> 本周的重要论文包括谷歌联合哈佛大学 Lichtman 实验室推出的「H01」数据集；中国科学技术大学郭光灿院士团队李传锋、周宗权研究组利用固态量子存储器和外置纠缠光源，首次实现两个吸收型量子存储器之间的可预报量子纠缠，演示了多模式量子中继等研究。

**目录：**

1. 3D AffordanceNet: A Benchmark for Visual Object Affordance Understanding
2. ACTION-Net: Multipath Excitation for Action Recognition 
3. A Connectomic Study of a Petascale Fragment of Human Cerebral Cortex 
4. Balance Control of a Novel Wheel-legged Robot: Design and Experiments
5. Heralded Entanglement Distribution between Two Absorptive Quantum Memories
6. Graph-Based Deep Learning for Medical Diagnosis and Analysis: Past, Present and Future
7. Enhancing Photorealism Enhancement

**论文 1：3D AffordanceNet: A Benchmark for Visual Object Affordance Understanding**

- 作者：Shengheng Deng、Xun Xu、Chaozheng Wu 等
- 论文地址：[https://arxiv.org/pdf/2103.16397.pdf](https://arxiv.org/pdf/2103.16397.pdf)

摘要：为了促进视觉功能可供性在真实场景中的研究，在这篇 CVPR 2021 论文中，来自华南理工大学等机构的研究者提出了基于 3D 点云数据的功能可供性数据集 3D AffordanceNet。基于此数据集，研究者提供了三个基准任务，用于评估视觉功能可供性理解。他们在所提出的 3D AffordanceNet 数据集基础上，提出了 3 个视觉功能可供性理解任务，并对利用半监督学习方法进行视觉功能可供性理解以利用未标注的数据样本的方式进行了探索，三个基线方法被用于在所有任务上进行评估，评估结果表明研究者提出的数据集和任务对视觉功能可供性理解在具有价值的同时，也具有挑战性。

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210606230043.png)

*3D AffordanceNet 数据集样例。*

推荐：本文已被 CVPR 2021 会议接收。

**论文 2：ACTION-Net: Multipath Excitation for Action Recognition**

- 作者：Zhengwei Wang、Qi She、Aljosa Smolic
- 论文地址：[https://arxiv.org/pdf/2103.07372.pdf](https://arxiv.org/pdf/2103.07372.pdf)

摘要：本文由字节跳动研究员佘琪和都柏林圣三一大学王正蔚合作完成，关注高效视频特征学习。视频应用场景近几年变得越来越多元化比如视频分类、视频精彩时刻挖掘和人机交互。在此工作中，主要侧重于时序动作识别比如人机交互与 VR /AR 中的手势识别。和传统的动作识别相比如 Kinetics（注重视频分类），此类应用场景主要有两种区别：其一是 一般部署在边缘设备上如手机和 VR / AR 设备上，所以对模型计算量和推理速度有一定的要求；其二此类动作（「Rotate fists counterclockwise」vs「Rotate fists clockwise」）和传统动作识别动作（「Walking」vs「Running」）相比有着较强时序性。针对以上的两点，基于 2D CNN（轻便）提出了一个混合注意力机制的 ACTION 模块（对于时序动作建模）。

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210606230127.png)

*ACTION 模块包含三个子模块，分别是时空激励（STE）模块、通道激励（CE）模块和运动激励（ME）模块。*

推荐：2D 和 3D CNN 解决不好视频动作识别，字节跳动等提出更高效的 Action 模块。

**论文 3：A Connectomic Study of a Petascale Fragment of Human Cerebral Cortex**

- 作者：Alexander Shapson-Coe、 Michał Januszewski、Daniel R. Berger 等
- 论文地址：[https://www.biorxiv.org/content/10.1101/2021.05.29.446289v1.full.pdf+html](https://www.biorxiv.org/content/10.1101/2021.05.29.446289v1.full.pdf+html)

摘要：谷歌联合哈佛大学 Lichtman 实验室于近日推出了「H01」数据集，这是一个 1.4 PB 的人类脑组织小样本渲染图。H01 样本通过连续切片电子显微镜获得了 4nm 分辨率的图像，利用自动计算技术进行重建和注释，并进行分析以初步了解人类皮层的结构。该项目的主要目标是为研究人脑提供一种新的资源，并改进和扩展潜在的连接组学技术。「H01」数据集包含了大约 1 立方毫米脑组织的成像数据，包括数以万计的重建神经元、数百万个神经元片段、1.3 亿个带注释的突触、104 个校对过的细胞，以及许多额外的亚细胞注释和结构，所有这些都可以通过 Neuroglancer 浏览器界面轻松访问。这是迄今为止人类编制的最全面、最详细的「人类大脑地图」，也是第一个大规模研究人类大脑皮层的突触连接的样本，该成果为研究人类大脑提供了重要资源。这一样本仍然只是整个人类大脑容量的百万分之一，未来的扩展研究仍然是一个巨大的技术挑战。

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210606230201.png)

推荐：1.3 亿突触、数万神经元，谷歌、哈佛发布史上最强三维「人脑地图」。

**论文 4：BalanceControlof aNovelWheel-leggedRobot:DesignandExperiments**

- 作者：ShuaiWang、LeileiCui2,∗,JingfanZhang 等
- 网盘链接：[https://pan.baidu.com/s/1S84x03gYg9YfMshndBBbPw](https://pan.baidu.com/s/1S84x03gYg9YfMshndBBbPw)
- 密码: 9qzz

摘要：今年 3 月 2 日，腾讯发布多模态四足机器人，引起了极大关注，今日，继 Max 之后，腾讯 Robotics X 实验室又一全新机器人亮相：轮腿式机器人 Ollie（奥利），它像一个灵活的「轮滑小子」，能完成跳跃、360 度空翻等高难度动作。伴随着 Ollie 的亮相，腾讯 Robotics X 实验室也公布了技术细节，相关论文已被 ICRA 2021 收录，介绍了轮腿式机器人平衡控制器的设计思路与实验结果。日前在西安举办的 ICRA 2021，腾讯 AI Lab 及 Robotics X 实验室主任张正友博士也受邀作大会报告，介绍了 Robotics X 实验室在机器人移动研究领域的布局与进展，并分享了 Ollie 的技术细节。

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210606230240.png)

推荐：跳跃、360 度空翻也能轻松搞定。

**论文 5：Heralded Entanglement Distribution between Two Absorptive Quantum Memories**

- 作者：Xiao Liu、Jun Hu、Zong-Feng Li
- 论文地址：[https://www.nature.com/articles/s41586-021-03505-3](https://www.nature.com/articles/s41586-021-03505-3)

摘要：当两个量子产生纠缠，一个变了，另一个也会瞬变，无论相隔多远，借助量子纠缠可实现量子通信。近期，中国科学技术大学郭光灿院士团队李传锋、周宗权研究组利用固态量子存储器和外置纠缠光源，首次实现两个吸收型量子存储器之间的可预报量子纠缠，演示了多模式量子中继。这是量子存储和量子中继领域的重大进展。中科院量子信息重点实验室的博士后刘肖和博士研究生胡军为该论文的共同第一作者。《Nature》杂志审稿人对该工作给予高度评价：「这是在地面上实现远距离量子网络的一项重大成就。」

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210606231003.png)

*原理示意图 。*

推荐：中科大再登 Nature 封面，郭光灿团队首次实现多模式量子中继。

**论文 6：Graph-Based Deep Learning for Medical Diagnosis and Analysis: Past, Present and Future**

- 作者：David Ahmedt-Aristizabal、Mohammad Ali Armin、Simon Denman 等
- 论文地址：[https://arxiv.org/pdf/2105.13137.pdf](https://arxiv.org/pdf/2105.13137.pdf)

摘要：在本文中，来自昆士兰科技大学和 CSIRO Data61 的研究者对图神经网络（GNN）模型在医疗诊断和分析方面的研究和进展做了全面回顾，其中解释 GNN 在该领域的重要性，强调了新的医疗分析挑战以及对未来工作的展望。

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210606230326.png)

*用于功能性连接的图卷积网络（GCN）方法及相关应用。*

推荐：最新「图深度学习医疗诊断与分析」综述论文，41 页 pdf319 篇文献。

**论文 7：Enhancing Photorealism Enhancement**

- 作者：Stephan R. Richter、Hassan Abu AlHaija、Vladlen Koltun
- 论文地址：[https://arxiv.org/abs/2105.04619](https://arxiv.org/abs/2105.04619)

摘要：近日，英特尔推出了一种深度学习系统，可将 3D 渲染图形转换为逼真的图片。侠盗猎车手 5（GTA 5）上进行测试时，该系统给出了令人印象深刻的结果。此前 GTA 5 的开发人员在重建洛杉矶和南加州的景观方面已经做得非常出色，现在借助英特尔的新系统，画面中的高质量合成 3D 图形能够变为现实生活的描绘。照片级渲染引擎处理单帧可能就要花费几分钟甚至几小时，而英特尔的新系统则能够以相对较高的帧速率处理图像。并且研究者表示，他们还将进一步优化该深度学习模型以更快地工作。这是否意味着实时逼真的游戏引擎即将出现？这很难说，因为还有几个基本问题尚未解决。为此他们撰写了一篇论文来描述该系统的性能，并与其他相似系统进行了对比实验。

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210606230400.png)

*左上为 GTA 中的 3D 渲染图，另外 3 幅为英特尔新模型生成结果。*

推荐：3D 渲染图变逼真图片，英特尔图像增强新模型将真实感拉满。

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210603140942.png)