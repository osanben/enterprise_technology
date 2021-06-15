# 7亿后台日志揭秘程序员如何面向Stack Overflow编程：获SIGSOFT杰出论文奖

> 你知道程序员是如何寻（fu）找（zhi）答（zhan）案（tie）的吗？

作为全世界最流行的编程问答网站，Stack Overflow 已经成为全世界程序员的福音，面向 Stack Overflow 编程成了程序员的必备技能。

在发表于全球软件工程年会 ICSE 2021 上的论文《Automated Query Reformulation for Efficient Search based on Query Logs From Stack Overflow》中，研究者通过分析 Stack Overflow 后台服务器中的超过 7 亿条日志，揭秘了程序员是如何寻（fu）找（zhi）答（zhan）案（tie）的，并提出了一种基于深度学习的自动化查询重构方法，该论文获得了 ACM SIGSOFT Distinguished Paper Award。

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210615093847.png)

论文地址：[https://arxiv.org/abs/2102.00826](https://arxiv.org/abs/2102.00826)

**大家什么时候在摸鱼？**

该研究使用的数据集包含 Stack Overflow 网站上 2017 年 12 月至 2018 年 11 月间的 7 亿多条 HTTP 请求。从月份维度来看，全年间网站的访问量基本保持稳定，年底的时候大家不得不为了 KPI 而（稍微）奋斗一些。

​    ![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210615093923.png)

以周为单位来看就非常 amazing 了，工作日用户的活动数量约为休息日活动数量的三倍，周六和周日没有太大的差别，看来大部分程序员都是忠实的 955 工作理念的践行者，那其他人呢？大小周？996？不存在的，他们都是 007。

 ![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210615093941.png)

**大家都在搜什么？**

随着 Python 编程语言近两年的火热，Python 理所当然地成为了最常搜索的关键词之一。编程语言、数据结构、API 名称等软件术语占据了用户查询词的大多数。

 ![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210615093958.png)

遇事不决就 how to，一言不合就贴错误日志，程序员的搜索方式你们 get 了吗？

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210615094014.png)

大部分的查询字符串的单词数都是很有限的，查询中包含的单词数的平均值为 3.6，中位数为 3，但也存在一定数量的超长查询，最长的甚至有 100 + 个单词，你猜猜他们都查了个啥？没错，错误日志。

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210615094031.png)

**高级查询你学废了吗？**

8.74% 的查询使用了高级查询语法，其中 “标签过滤”、“用户过滤” 和“特殊短语声明”三项占据了总数量的 93% 以上。由于 “用户过滤” 是在用户点击 profile 的时候自动触发的，因此可以排除在外，那么用户最常用的高级查询也就是标签过滤了，通配符和多重标签等比较复杂的语法规则大部分用户基本不会使用。

Stack Overflow 提供的高级查询语法规则：[https://stackoverflow.com/help/searching](https://stackoverflow.com/help/searching)

​    ![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210615094055.png)

**搜不到结果怎么办？**

当无法检索到满意结果时，程序员一般来说是绝望的，这报错解决不了今天是不想睡觉了吗？DDL（最后期限）马上就到了，能怎么办？改呗，把查询词修改下，再看看能不能查到想要的问题。

怎么改？这是一个好问题，一起来看看程序员是怎么做的吧。

该研究将修改查询的模式（查询重构模式）分为了增加、修改和删除三个类别，其中每个大类又细分成 2~3 个小类。不难看出增加编程语言或平台限制、拼写与语法检查、需求细化是最常见的查询重构模式。

​    ![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210615094119.png)

**修改查询时，改动量大吗？**

用户修改查询的时候是只会修改少数的单词或字符，还是会更换整个查询的表达方式呢？该研究对查询重构的修改幅度的实证研究结果显示，在 58.07% 的样本中，原查询与重构后的查询的相似度都大于 0.7，修改涉及的字符数量仅约等于一个查询词的字符数。

 ![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210615094132.png)

**自动执行查询重构**

基于实证研究的结果，该研究认为软件领域的查询重构模式众多，通过设计基于规则的启发式方法来实现软件领域的查询重构费时且容易出错，相反不涉及大幅修改的查询重构可以通过深度学习模型来建模。该论文提出了一种基于 Transformer 的软件领域查询重构方法 SEQUER。

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210615094158.png)

SEQUER 首先基于启发式规则从用户的活动日志中抽取查询重构序列，并构造查询重构对，然后使用这些语料训练了一个基于 Transformer 的模型，在完成模型训练后，当给定原查询，模型可以直接输出重构后的查询，相较于原查询，该重构后的查询可以更好地检索出用户满意的查询结果。

通过与五种最新基准方法的比较，SEQUER 给出的查询重构结果不仅更接近用户的手工重构，而且在检索用户满意的帖子任务上具有更好的性能。

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210615094216.png)

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210615094232.png)

对查询重构结果的深入分析结果显示，SEQUER 能够实现很多基准方法无法实现的查询重构模式，包括纠正错误的拼写，例如修正 how to import bumpy array 中的 bumpy 为 numpy；为查询增加编程语言或平台限制，例如从 requests negotiate 到[python] requests negotiate；删除查询中的特异信息，例如 truncated for column 'status'中的 status；用文字代替符号，例如从 a* search 到 a star search。

为了方便开发人员使用该论文提出的查询重构方法，研究者设计并上线了一款软件领域的查询重构插件，该插件可以为用户的查询生成 10 个候选的查询重构结果。

​    ![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210615094303.png)

插件网址：[https://github.com/kbcao/sequer](https://github.com/kbcao/sequer)

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina/20210615093836.png)