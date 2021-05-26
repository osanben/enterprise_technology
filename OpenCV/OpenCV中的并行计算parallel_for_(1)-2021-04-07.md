OpenCV中的parallel_for_为用户提供了快速并行化代码的途径。

先来看一个用parallel_for_并行化矩阵相乘的简单例子。这个例子是对输入为128x128x32进行核为3x3x64的卷积操作。

直接计算的代码如下：

```
void conv3x3(const Mat& inp, const Mat& weights, Mat& out)
{
    int out_cn = weights.size[0];
    int inp_cn = weights.size[1];
    int kh = weights.size[2], kw = weights.size[3];
    int inp_h = inp.size[1], inp_w = inp.size[2];
    int out_h = inp_h - kh + 1, out_w = inp_w - kw + 1;
    int size[] = {out_cn, out_h, out_w};

    CV_Assert( inp.size[0] == inp_cn && kw == 3 && kh == 3 );

    out.create(3, size, CV_32F);

    for( int oc = 0; oc < out_cn; oc++ )
        for( int y = 0; y < out_h; y++ )
        {
            int iidx[] = { 0, y, 0 };
            int widx[] = { oc, 0, 0, 0 };
            int oidx[] = { oc, y, 0 };

            const float* iptr0 = inp.ptr<float>(iidx);      // &inp[0, y, 0]
            const float* wptr0 = weights.ptr<float>(widx);  // &weights[oc, 0, 0, 0]
            float* optr = out.ptr<float>(oidx);             // &out[oc, y, 0]

            for( int x = 0; x < out_w; x++ )
            {
                float sum = 0.f;
                for( int ic = 0; ic < inp_cn; ic++ )
                {
                    const float* iptr = iptr0 + x + ic*(inp_h*inp_w);   // &inp[ic, y, x]
                    const float* wptr = wptr0 + ic*(kw*kh);             // &weights[oc, ic, 0, 0]

                    sum += iptr[0]*wptr[0] + iptr[1]*wptr[1] + iptr[2]*wptr[2] +
                           iptr[inp_w]*wptr[3] + iptr[inp_w+1]*wptr[4] + iptr[inp_w+2]*wptr[5] +
                           iptr[inp_w*2]*wptr[6] + iptr[inp_w*2+1]*wptr[7] + iptr[inp_w*2+2]*wptr[8];
                }
                optr[x] = sum;
            }
        }
}
```

使用parallel_for_进行并行计算的代码如下：

```
void conv3x3_parallel(const Mat& inp, const Mat& weights, Mat& out)
{
    int out_cn = weights.size[0];
    int inp_cn = weights.size[1];
    int kh = weights.size[2], kw = weights.size[3];
    int inp_h = inp.size[1], inp_w = inp.size[2];
    int out_h = inp_h - kh + 1, out_w = inp_w - kw + 1;
    int size[] = {out_cn, out_h, out_w};

    CV_Assert( inp.size[0] == inp_cn && kw == 3 && kh == 3 );

    out.create(3, size, CV_32F);

    // 用parallel_for_按ch进行并行计算
    parallel_for_(Range(0, out_cn), [&](const Range& r)
    {
    for( int oc = r.start; oc < r.end; oc++ )
        for( int y = 0; y < out_h; y++ )
        {
            int iidx[] = { 0, y, 0 };
            int widx[] = { oc, 0, 0, 0 };
            int oidx[] = { oc, y, 0 };

            const float* iptr0 = inp.ptr<float>(iidx);      // &inp[0, y, 0]
            const float* wptr0 = weights.ptr<float>(widx);  // &weights[oc, 0, 0, 0]
            float* optr = out.ptr<float>(oidx);             // &out[oc, y, 0]

            for( int x = 0; x < out_w; x++ )
            {
                float sum = 0.f;
                for( int ic = 0; ic < inp_cn; ic++ )
                {
                    const float* iptr = iptr0 + x + ic*(inp_h*inp_w);   // &inp[ic, y, x]
                    const float* wptr = wptr0 + ic*(kw*kh);             // &weights[oc, ic, 0, 0]

                    sum += iptr[0]*wptr[0] + iptr[1]*wptr[1] + iptr[2]*wptr[2] +
                    iptr[inp_w]*wptr[3] + iptr[inp_w+1]*wptr[4] + iptr[inp_w+2]*wptr[5] +
                    iptr[inp_w*2]*wptr[6] + iptr[inp_w*2+1]*wptr[7] + iptr[inp_w*2+2]*wptr[8];
                }
                optr[x] = sum;
            }
        }
    });
}
```

来运行一下

```
int main(int argc, char** argv)
{
    const int inp_h = 128, inp_w = 128, inp_cn = 32;
    const int out_cn = 64;
    const int kh = 3, kw = 3;

    Mat inp, w, out_ref, out_curr;
    gen_inp(inp_cn, inp_h, inp_w, inp);
    gen_weights(out_cn, inp_cn, kh, kw, w);

    conv3x3(inp, w, out_ref);
    conv3x3(inp, w, out_curr);
    double t = (double)getTickCount();
    conv3x3(inp, w, out_curr);
    t = (double)getTickCount() - t;

    double t2 = (double)getTickCount();
    conv3x3_parallel(inp, w, out_curr);
    t2 = (double)getTickCount() - t2;

    printf("conv3x3 time = %.1fms\n", t * 1000 / getTickFrequency());
    printf("conv3x3_parallel time = %.1fms\n",  t2 * 1000 / getTickFrequency());

    return 0;
}    
```

conv3x3和conv3x3_parallel在我的笔记本电脑上运行时间对比如下：

![](./imgs/7.png)

对比conv3x3和conv3x3_parallel的内部实现，基本相同，只是conv3x3_parallel代码中多了一句用parallel_for_按照输出通道的数量进行并行计算。parallel_for_根据用户计算机上的并行框架在其内部完成了代码的并行化，非常简便易用！

使用parallel_for_的一个前提条件是OpenCV需要与并行框架一起编译。OpenCV中支持以下并行框架，并按照下面的顺序选取进行处理：

- Intel TBB (第三方库，需显式启用)
- C=并行C/C++编程语言扩展 (第三方库，需显式启用)
- OpenMP (编译器集成, 需显式启用)
- APPLE GCD (苹果系统自动使用)
- Windows RT并发(Windows RT自动使用)
- Windows并发(运行时部分, Windows，MSVC++ >= 10自动使用)
- Pthreads 

在刚发布的OpenCV 4.5.2版本，增加了支持并行框架的选择。特殊编译的OpenCV可以允许选择并行后端，并/或通过plugin动态载入。如：

```
# TBB plugin
cd plugin_tbb
cmake <opencv>/modules/core/misc/plugins/parallel_tbb
cmake --build . --config Release

# OpenMP plugin
cd plugin_openmp
cmake <opencv>/modules/core/misc/plugins/parallel_openmp
cmake --build . --config Release
```

第二个条件是所要完成的计算任务是适宜并可以进行并行化的。简言之，可以分解成多个子任务并且没有内存依赖的就容易并行化。例如，对一个像素的处理并不与其他像素的处理冲突，计算机视觉相关的任务多数情况下容易进行并行化。

参考资料：

[1] [https://docs.opencv.org/master/d7/dff/tutorial_how_to_use_OpenCV_parallel_for_.html](https://docs.opencv.org/master/d7/dff/tutorial_how_to_use_OpenCV_parallel_for_.html)

[2] [https://github.com/opencv/opencv/wiki/ChangeLog](https://github.com/opencv/opencv/wiki/ChangeLog)

[3] [https://github.com/opencv/opencv/pull/19470](https://github.com/opencv/opencv/pull/19470)