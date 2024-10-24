# simple-CBIR-core
a simple CBIR(Content-based image retrieval) system's core, based on pochih's CBIR. 

一个简易的CBIR系统（基于内容的图像检索）核心，基于pochih的CBIR项目.

本项目是一个基于[pochih](https://github.com/pochih)所写的[CBIR](https://github.com/pochih/CBIR)所修改的一个项目。

由于源项目所实现的功能与我所需实现的功能差别有点大，故在fork的基础上重新开了一个项目。

本项目在源项目的基础上加入了部分更为实用的接口，并且加入了对数据库的增删改查的接口，当然，目前还在修改中。

故本项目还在施工中...

# 简易使用方法：

```
from vggnet import VGGNetFeat
d = OneData(img_path)  # img_path 为所需要以图搜图的图片路径
vgg = VGGNetFeat()
t = vgg.extract_features(d)[0]    # 用extract_features()提取图片的特征向量
prod = infer_dis(t,db=db, sample_db_fn=vgg.make_samples,return_img=True)  # 使用infer_dir()获取该图片与数据库中图片的距离
```

接下去的适合在有缓存文件时使用：

inder_dis()可以直接传缓存文件进去，例如 ``` prod = infer_dis(t, samples, return_img=True) ``` 其中samples是缓存文件，缓存文件为二进制文件，使用 ```pickle.load(open(file, 'rb'))``` 打开。

获取缓存文件可以使用最上面那个例子的方法，也可以用 ```chage2 = creat_feature(db, f_class=VGGNetFeat)``` 其中f_class 是所用的特征提取方法所在的类。

那么那么好用的方法是谁写的呢，是原作者写的。我本来以为原作者没写对外的接口，仔细看了下原作者好像是写了的。



# 开发日志？

2024.10.18: 

在 DB.py 中新建了一个类 OneData 该类用于接收单张临时输入图片。该类继承自BD中的 Database 类。新建该类是用于在外部图片与本地数据库中的图片进行比较使用，也就是以图搜图时的输入图片。

在 vggnet.py 的 VGGNetFeat 类中新建了一个方法 extract_fratures，该方法是从 make_samples 方法中拆分出来的，这是因为在对单张图进行以图搜图输入时需要对这张输入的图进行特征提取。

在 evaluate.py 的 infer 函数中添加了形参 return_img，该参数用于控制输出的向量距离组（即输入图片的向量与库中图片的向量的距离的集合）中库中图片的本地位置。该参数默认为False。


2024.10.24:

取消了该项目对py2的支持，因为我的开发环境没有py2，本项目只对自己的开发环境负责！

在 evaluate.py 中新建了一个函数 creat_feature，从 evaluate_class 中拆分出来，这样可以仅生成特征向量而不求其准度


