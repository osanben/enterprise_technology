本次教程的目的是带领大家学会用 Tensorflow serving 部署训练好的模型

这里我们用到的数据集是 Fashion MNIST，所以训练出来的模型可以实现以下几个类别的分类

```python
'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
```

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210725172235.png)

因为这篇教程主要关注部署，所以我们直接从已经训练好的模型开始，保存的格式是 SavedModel，如上图所示

在这之前呢，我们需要先安装好 tensorflow_model_server

接下来我们可以在控制台执行以下指令，就可以启动一个 serving 服务了，我们可以通过 REST API 进行请求，并返回预测结果

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210725172452.png)

```
import requests
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/fashion_mnist:predict', data=data, headers=headers)

predictions = json.loads(json_response.text)["predictions"]

show(0, "The model thought this was a {} (class {}), and it was actually a {} (class {})".format(class_names[np.argmax(predictions[0])], np.argmax(predictions[0]), class_names[test_labels[0]], test_labels[0]))
```

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210725172649.png)

上图是通过请求，然后预测得到的结果，到此，我们实现了模型的 Tensorflow serving 的部署

代码链接: https://codechina.csdn.net/csdn_codechina/enterprise_technology/-/blob/master/tensorflow_serving.ipynb