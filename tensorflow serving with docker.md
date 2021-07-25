本次教程的目的是带领大家看下如何用 Docker 部署深度学习模型的

第一步我们需要 pull 一个 docker image

```bsh
sudo docker pull tensorflow/serving
```

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210725223155.png)

如上图所示，执行 pull 之后，我们看到本地已经存在 tensorflow/serving:latest

接下来我们 clone 一个仓库

```bsh
git clone https://github.com/tensorflow/serving
```

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210725223402.png)

上图中的 saved_model_half_plus_two_cpu 就是我们想要部署的模型

然后我们可以直接运行以下命令实现部署

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210725223549.png)

运行结果如下图所示，我们可以看到 Exporting HTTP/REST API at:localhost:8501，那么就代表着部署成功了

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210725223630.png)

接下来我们可以进行预测，返回的结果也能对的上

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/20210725223742.png)