# Git版本回退

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/2021071510:04:59.png)

1. 创建一个practice_git文件夹
2. 第二步执行git init，初始化一个由git进行版本控制的仓库，从图中可以看到，我先运行了git status，结果报错，说不存在git仓库
3. 用touch命令创建readme.txt
4. 运行git status命令，可以看到目前readme.txt是untracked file
5. 执行git add readme.txt把readme.txt文件添加到暂存区
6. 执行git commit命令提交readme.txt

上面几个步骤我们完成了仓库初始化，文件的创建、缓存、提交

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/2021071510:49:48.png)

下面我们要对文件进行更改，更改完之后可执行git diff查看修改，如上图所示，我们可以看到添加了一行add content to this file，以及一个说明changes not staged for commit，接下来我们可以运行git add readme.txt，然后再执行git commit进行提交修改

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/2021071510:55:10.png)

到目前为止，我们一共进行了2次提交，接下来执行git log指令查看提交历史记录，如上图所示

27cd073dcf66a50658d98cf0f7e60e51028292ab，你看到的这么一大串是commit id

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/2021071511:01:50.png)

运行git reset --hard 496d9f61f8之后，我们发现确实回退到上一个版本了，但是现在我们看不到之前的那个版本了，

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/2021071511:25:11.png)

此时我们可以看到git reflog，发现27cd073 HEAD@{1}: commit: add content to this file这么一行，如果现在想回到最新版本，可以运行git log来查看

![](https://maoxianxin1996.oss-accelerate.aliyuncs.com/codechina1/2021071511:26:51.png)

从上图我们可以发现，已经成功的回到了最新版本