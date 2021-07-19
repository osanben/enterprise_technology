## 首先我们要讲一下为什么提到 Tensorflow

TensorFlow 是一个端到端的机器学习开源平台。它有一个全面、灵活的工具，函数库和社区资源生态系统，可以让研究人员推动 ML 领域的最先进技术，以及让开发人员轻松构建和部署 ML 驱动的应用程序

从上面这段话，我们可以看出 Tensorflow 有一个非常强的生态系统，可以让开发者很轻松的走完整条项目开发链，这让它在工业界的应用方面占有非常大的优势，同时由于全面、灵活的工具、函数库，这让研究人员可以很方便的使用它来做各种实验，综上我们可以知道 Tensorflow 为什么在学术界和工业界有很多的用户群体

## 接下来我们一起看下 Tensorflow 有什么特性呢

**简单的模型构建**

使用直观的高级 API (如 Keras )快速执行，轻松地构建和训练 ML 模型，这使得模型的快速迭代和调试变得容易。

**在任何地方都能实现强大的 ML 生产**

无论您使用何种语言，都可以轻松地在云、在线、浏览器或设备上训练和部署模型。

**强有力的实验研究**

一个简单而灵活的体系结构，从概念到代码，到最先进的模型，并更快地发布新思想。

## 我们再来看下 Tensorflow 可以用在什么地方

**Tensorflow**

帮助您开发和训练 ML 模型的核心开源库。可以直接在浏览器中运行 Colab 笔记本，快速入门。

**For JavaScript**

TensorFlow.js 是一个 JavaScript 库，用于在浏览器和 Node.js 上训练和部署模型。

**For Mobile & IoT**

TensorFlow Lite 是一个轻量级库，用于在移动设备和嵌入式设备上部署模型。

**For Production**

TensorFlow Extended 是一个端到端平台，用于在大型生产环境中准备数据、训练、验证和部署模型。

## 最后我们来看一下针对初学者和专家的不同代码模板

### For beginners

最好从用户友好的 Sequential API 开始。您可以通过将构建块插入到一起来创建模型。运行下面的 Hello World 示例，然后访问教程了解更多信息

```
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

上面是一个简单的代码片段

### For experts

Subclassing API 为高级研究提供了一个按运行定义的接口。为您的模型创建一个类，然后编写命令式的前向传递。轻松地编写自定义层、激活和训练循环。运行下面的 Hello World 示例，然后访问教程了解更多信息

```
class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)
model = MyModel()

with tf.GradientTape() as tape:
  logits = model(images)
  loss_value = loss(logits, labels)
grads = tape.gradient(loss_value, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

上面是一个简单的代码片段

最后放一个Tensorflow代码仓库: https://codechina.csdn.net/mirrors/tensorflow/tensorflow