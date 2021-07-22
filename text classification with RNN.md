本教程的目的是带领大家学会用 RNN 进行文本分类

本次用到的数据集是 IMDB，一共有 50000 条电影评论，其中 25000 条是训练集，另外 25000 条是测试集

首先我们需要加载数据集，可以通过 TFDS 很简单的把数据集下载过来，如下代码所示

```
dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']

train_dataset.element_spec
```

接下来我们需要创建 text encoder，可以通过 tf.keras.layers.experimental.preprocessing.TextVectorization 实现，如下代码所示

```
VOCAB_SIZE = 1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE
)
encoder.adapt(train_dataset.map(lambda text, label: text))
```

接下来我们需要搭建模型，下图是模型结构图

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210722205501.png)

对应的代码如下所示

```
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
```

到这一步，我们就可以开始训练了，以及训练后进行模型评估

```
history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30)

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)
```

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210722205743.png)

上面是训练的结果记录图

代码地址: https://codechina.csdn.net/csdn_codechina/enterprise_technology/-/blob/master/text_classification_rnn.ipynb