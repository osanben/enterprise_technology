本次教程的目的是实现 Fashion MNIST 分类，之前我们提到过 MNIST ，你可以把 Fashion MNIST 看成是 MNIST 的替代，为什么呢？

主要是样本的多样性以及 Fashion MNIST 比 MNIST 更具有挑战性

接下来我们一起看看 Fashion MNIST 数据集长什么样，如下图所示

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210720150053.png)

该数据集有 70000 张灰度图片，一共有 10 个类别，图片的像素是 28x28 ，分辨率较低，其中 60000 张图片是训练集，剩下的 10000 张图片是测试集，用于评估模型训练结果

接下来我们导入需要的函数库，以及加载数据集

```
import tensorflow as tf
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0
```

加载上来的图片是一个 28x28 的 Numpy 数组，像素值是 0-255 ，对应的标签值是 0-9 ，下表是标签值对应的类别名

| Label | Class       |
| :---- | :---------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

接下来是网络模型搭建

```
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

我们具体讲一下以下 3 个点:

1. Loss function -- 我们的训练目标是，让这个值越小越好
2. Optimizer -- 用于更新模型参数，基于输入数据和损失值
3. Metrics -- 用于跟踪训练和测试过程中的损失值和准确率

通过调用 model.fit() ，开启模型训练，以及调用 model.evaluate() 进行训练后模型评估

```
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
```

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210720151805.png)

如上图所示，我们对 test_images ，进行了预测，并输出第一张图片的预测结果和实际标签，发现两者一致

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210720151912.png)

接下来我们对单张图片进行预测，在这之前我们需要 add 一个维度，使图片的 shape 从 (28, 28) 变成 (1, 28, 28) ，我们可以发现预测结果和实际标签也一致

最后分析一下训练集上的准确率和测试集上的准确率的Gap原因？

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210720153736.png)

如上图所示，train_acc=0.91，test_acc=0.88，0.88 < 0.91 并且差距不小，造成这个的原因是过拟合，意思就是模型在训练集上的表现比测试集上的好，有一部分原因是模型记住了训练集的数据和其中的noise，从而对新数据的预测造成负面影响

代码地址: https://codechina.csdn.net/csdn_codechina/enterprise_technology/-/blob/master/Basic%20classification:%20Classify%20images%20of%20clothing.md