## 如何运行代码

我们看看如何使用这些代码

它可以分为两部分

### 1. 创建你的图像数据库
当你clone这个项目后，它看上去是这样的:

    ├── src/            # 源文件
    ├── result/         # 结果
    ├── USAGE.md        # 如何使用代码
    └── README.md       # 项目的介绍

你需要将你的图像放在文件夹 __database/__ 内, 所有它看起来应该是这样的:

    ├── src/            # 源文件
    ├── result/         # 结果
    ├── USAGE.md        # 如何使用代码
    ├── README.md       # 项目的介绍
    └── database/       # 你的图片的文件夹

__你所有的图片都要放在 database/ 中__

在这个文件夹内，每个图像类都应该有自己的文件夹

参考如下图片:

<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='https://github.com/pochih/CBIR/blob/img/database.png' padding='5px'></img>

在我的数据库中有25个类, 每个类都有自己的文件夹

属于这个类的图像应该归入这个文件夹中.

### 2. 运行代码
我实现了这几个算法，你可以用python3运行它

#### RGB直方图 For RGB histogram
```python
python3 src/color.py
```

#### Daisy局部图像特征描述子 For daisy image descriptor
```python
python3 src/daisy.py
```

#### Gabor滤波器 For gabor filter
```python
python3 src/gabor.py
```

#### 边缘直方图 For edge histogram
```python
python3 src/edge.py
```

#### 梯度直方图 For histogram of gradient (HOG)
```python
python3 src/HOG.py
```

#### VGG19模型 For VGG19
需要使用 pytorch0.2 
```python
python3 src/vggnet.py
```

#### ResNet152模型 For ResNet152
需要使用 pytorch0.2 
```python
python3 src/resnet.py
```

以上是代码的基本用法.

有些高级的方法，例如特征融合与降维,

这些部分我们将在之后编写 :D

### 附录: 特征融合
我实现了基本的特征融合方法 -- 拼接 concatenation.

特征融合的代码是 [fusion.py](https://github.com/pochih/CBIR/blob/master/src/fusion.py)

在 fusion.py 中，有个名为 *FeatureFusion* 的类.

你可以使用名为 **features** 的参数创建 *FeatureFusion* 实例.

例如, 在 [fusion.py line140](https://github.com/pochih/CBIR/blob/master/src/fusion.py#L140)
```python
fusion = FeatureFusion(features=['color', 'daisy'])
APs = evaluate_class(db, f_instance=fusion, d_type=d_type, depth=depth)
```
- 第一行表示连接颜色特征 color feature 与daisy特征 daisy feature.
- 第二行表示使用串联特征进行评估.

如果你想了解所有可能的特征组合的性能, 请参考 [fusion.py line122](https://github.com/pochih/CBIR/blob/master/src/fusion.py#L122) 
```python
evaluate_feats(db, N=2, d_type='d1')
```
- 参数 *N* 表示要连接的特征数量.
- 参数 *d_type* 表示要使用的距离度量.
- 函数 *evaluate_feats* 将会生成一个结果文件，用于记录所有特征串的性能.

## 作者
Po-Chih Huang / [@pochih](http://pochih.github.io/)
### 翻译
Dedicatus1979 / [@pochih](https://github.com/Dedicatus1979/)
