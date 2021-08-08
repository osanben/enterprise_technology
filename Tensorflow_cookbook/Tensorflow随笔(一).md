In machine learning, to improve something you often need to be able to measure it. TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow. It enables tracking experiment metrics like loss and accuracy, visualizing the model graph, projecting embeddings to a lower dimensional space, and much more.

Using the MNIST dataset as the example, normalize the data and write a function that creates a simple Keras model for classifying the images into 10 classes.

```
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
```

When training with Keras's Model.fit(), adding the tf.keras.callbacks.TensorBoard callback ensures that logs are created and stored. Additionally, enable histogram computation every epoch with histogram_freq=1 (this is off by default)

Place the logs in a timestamped subdirectory to allow easy selection of different training runs.

```
model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x=x_train, 
          y=y_train, 
          epochs=5, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])
```

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210807200805.png)

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210807200823.png)

A brief overview of the dashboards shown (tabs in top navigation bar):

- The **Scalars** dashboard shows how the loss and metrics change with every epoch. You can use it to also track training speed, learning rate, and other scalar values.
- The **Graphs** dashboard helps you visualize your model. In this case, the Keras graph of layers is shown which can help you ensure it is built correctly.
- The **Distributions** and **Histograms** dashboards show the distribution of a Tensor over time. This can be useful to visualize weights and biases and verify that they are changing in an expected way.

Additional TensorBoard plugins are automatically enabled when you log other types of data. For example, the Keras TensorBoard callback lets you log images and embeddings as well. You can see what other plugins are available in TensorBoard by clicking on the "inactive" dropdown towards the top right

x_train.shape = (60000, 28, 28)

min = 0

max = 255



y_train.shape = (60000,)

min = 0

max = 9



```
x_train, x_test = x_train / 255.0, y_test / 255.0
对数据进行MinMaxScaler()，缩放到[0,1]
作用:
	加快学习算法的收敛速度
	使不同量纲的特征处于同一数值量级，减少方差大的特征的影响，使模型更准确
```