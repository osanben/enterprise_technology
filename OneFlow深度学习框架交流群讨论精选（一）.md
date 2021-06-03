# OneFlow深度学习框架交流群讨论精选（一）

OneFlow 即将开源，“OneFlow 深度学习框架微信交流群”中已经聚集了一大群深度学习理论研究者、工程实践者、知识精英，在OneFlow 开源前讨论对 OneFlow 的期待，以及深度学习框架的未来趋势。 

不少讨论的技术话题我们认为很有价值，我们选择其中有代表性的展示给未能参与讨论的朋友们。

以下内容摘录自 2020.7.25 OneFlow深度学习框架交流群。

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210603161645.png)

**太长不看版本**

举个例子，整个神经网络有的层次在一组卡上，另外一些层次在另一组卡上，两组卡以接力的方式协同工作。谷歌有一篇文章 gpipe。是分多个阶段，在设备之间流水执行。 

OneFlow团队通过理论分析证明了在某些特定场景下，流水并行是最优选择，并在OneFlow中应用。

**讨论过程：**

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210603161708.png)

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210603161749.png)

**太长不看版本**

模型并行的难度主要在于将模型切分到具体的物理设备上，编程和调试难度都较高，其次模型并行中高效率地实现也很难。 

TensorFlow和Pytorch因为历史包袱的原因，在已有框架下做模型并行，较难有优雅且高效的实现。 

个别后发的厂商，提出了自己的方案解决模型并行问题，包括OneFlow。

**讨论过程：**

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210603161813.png)

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210603161837.png)

**太长不看版本**

虽然有越来越多的超大规模模型面世，但是不能断定大模型是趋势。已有框架对大规模模型问题的支持并不理想，往往需要定制框架。OneFlow 想从框架级别解决这类问题，并且认为解决问题的过程中积累的经验，对于非大规模模型问题，也是有益的。 

想从软件角度解决深度学习的算力问题，让大量“一般”的芯片协同起来像一个“超级芯片”那样工作，让分布式训练中的“核武器”平民化。

**讨论过程：**

(有人举了BERT、GPT等大模型例子) 

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210603161857.png)

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210603161923.png)

**太长不看版本**

为了方便用户从其它框架到OneFlow的转入转出。OneFlow支持onxx，在一些固定结构的模型上可以直接转化，相关工作还在持续开发。 

OneFlow在开源同时，开放的Model Zoo中会包括一些常见的模型，它们与pytorch、tensorflow均已对齐。预训练模型也会逐步完善。 

与其它框架对标的常见op均已提供，部分少见的op也在完善，并且可以让用户自定义op。

**讨论过程：**

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210603161941.png)

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210603140942.png)