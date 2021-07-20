本次教程的目的是快速带领初学者入门以及使用 Tensorflow 2 ，一共分为以下 3 个知识点:

1. 搭建一个神经网络用于图片分类
2. 训练搭建好的神经网络
3. 对训练好的网络模型进行准确率评估

首先我们需要 Import 用到的函数库

```
import numpy as np
import tensorflow as tf
```

加载 MNIST 数据集，该数据集相当于编程界的 "Hello, World" ，然后把样本的数据类型从 int 转化成 float

```
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

如下图所示，我们可以看到加载上来的数据集的数据类型为 uint8 ，取值范围是 0-255 ，对应了图片的像素值 0-255

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210720112034.png)

转化之后，如下图所示，变成了 float64

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210720112116.png)

接下来我们进入模型搭建阶段，用的是 Sequential API ，同时我们还需要选择优化器和损失函数以用于训练

```
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
```

注解: 针对上面模型搭建阶段，为什么不把 `tf.nn.softmax` 放入最后一层 `Dense` 的原因是，当使用 softmax 输出时，不可能为所有模型提供精确和数值稳定的损失计算

到这一步，我们就可以开始训练模型了

```
model.fit(x_train, y_train, epochs=5)
```

训练完模型之后，我们还需要对模型的准确率进行评估

```
model.evaluate(x_test, y_test, verbose=2)
```

最后我们对测试集的 5 张图片进行预测，看看模型是否真的学会图片分类了

```
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

print(np.argmax(probability_model(x_test[:5]), axis=1))
print(y_test[:5])
```

一般来说，预测结果和标签是能够一一对应上的

一些补充:

1. 我们可以测试一下模型在未训练前的输出
2. 我们可以测试一下模型在训练后的输出

## 模型在未训练前的输出

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210720121724.png)

上图展示的是，模型未训练前，计算出来的损失值，这背后有什么奥秘呢？

首先我们要知道一点，MNIST 有 0-9 ，10 个数字，也就是 10 分类问题，未训练的模型对于每个类别预测正确的概率是 1/10 ，然后损失值计算公式是: `-tf.math.log(1/10) ~= 2.3` 和我们这里的 2.68 输出非常近似

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210720122113.png)

接下来我们输出一下模型未训练状态下的预测结果，可以看到 2/20=1/10 的正确率，基本符合随机预测的概率(当然这里统计的样本数过少，定义不够严谨)

## 模型在训练后的输出

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210720122332.png)

我们看到 evaluate 的输出是`[0.07140578329563141, 0.9786999821662903]` 前者是损失值，后者是准确率，我们发现经过 epoch=5 次迭代，模型在测试集上取得了 97.86% 的准确率

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210720122251.png)

上图是对测试集的前 5 张图片进行预测，可以看到预测结果均正确

代码地址: https://codechina.csdn.net/csdn_codechina/enterprise_technology/-/blob/master/tf2_quickstart_for_beginners.ipynb