[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source-150x25.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

## 简介 
__该项目是一个CBIR（基于图像的内容搜索）系统__

__提取图像的特征，并从图像库中检索相似的特征__

<img align='center' style="border-color:gray;border-width:2px;border-style:dashed"   src='https://github.com/pochih/CBIR/blob/img/CBIR.png' padding='5px' height="300px"></img>
<h6><a href='https://winstonhsu.info/2017f-mmai/'>Image src</a></h6>


## 1. 特征提取

在此系统，我实现了几个流行的特征识别功能:
- 基于颜色的特征识别
  - [RGB histogram](https://github.com/pochih/CBIR/blob/master/src/color.py)
- 基于纹理的特征识别
  - [gabor filter](https://github.com/pochih/CBIR/blob/master/src/gabor.py)
- 基于形状的特征识别
  - [daisy](https://github.com/pochih/CBIR/blob/master/src/daisy.py)
  - [edge histogram](https://github.com/pochih/CBIR/blob/master/src/edge.py)
  - [HOG (histogram of gradient)](https://github.com/pochih/CBIR/blob/master/src/HOG.py)
- 基于深度学习方式的特征识别
  - [VGG net](https://github.com/pochih/CBIR/blob/master/src/vggnet.py)
  - [Residual net](https://github.com/pochih/CBIR/blob/master/src/resnet.py)

##### *所有的功能均已模块化*

### 特征融合
部分特征不够健壮，并将其转为特征融合
- [fusion.py](https://github.com/pochih/CBIR/blob/master/src/fusion.py)

### 降维
维度灾难(Curse of Dimensionality)告诉我们，高维属性有时会失去距离特性
- [Random Projection](https://github.com/pochih/CBIR/blob/master/src/random_projection.py)



## 2. 评估

CBIR系统基于 __特征相似性__ 检索图像

系统的健壮性由MMAP (mean MAP)(Mean Average Precision 全类别平均正确率)进行评估, 评估方式参考 <a href='http://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-1-per.pdf' target="_blank">这里</a>

- image AP   : 每次命中的平均精度
  - depth=K 表示系统将返回前K张图片（与待搜索图片相似度最高的K张图片）
  - 在前K张图片中每有一张正确的图片称为一次命中
  - AP = (1次命中率 + 2次命中率 + ... + H次命中率) / H
- class1 MAP = (class1.img1.AP + class1.img2.AP + ... + class1.imgM.AP) / M
- MMAP       = (class1.MAP + class2.MAP + ... + classN.MAP) / N

评估的实现在 [evaluate.py](https://github.com/pochih/CBIR/blob/master/src/evaluate.py) 中找到

我的数据库中包括25个类, 每个类中20张图片, 总共500张图片, depth=K will 是从返回数据中返回前K张图片

Method | color | daisy | edge | gabor | HOG | vgg19 | resnet152
--- | --- | --- | --- |--- |--- |--- |---
Mean MAP (depth=10) | 0.614 | 0.468 | 0.301 | 0.346 | 0.450 | 0.914 | 0.944



## 3. 图像检索（每种方法返回前5个）
让我展示下该系统的一些结果

### query1 - women dress
#### query <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query1-women_dress/query1.jpg' padding='5px' height="80px"></img>
#### color <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query1-women_dress/query1-color.jpg' padding='5px' height="80px"></img>
#### daisy <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query1-women_dress/query1-daisy.jpg' padding='5px' height="80px"></img>
#### edge <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query1-women_dress/query1-edge.jpg' padding='5px' height="80px"></img>
#### gabor <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query1-women_dress/query1-gabor.jpg' padding='5px' height="80px"></img>
#### HOG <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query1-women_dress/query1-hog.jpg' padding='5px' height="80px"></img>
#### VGG19 <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query1-women_dress/query1-vgg.jpg' padding='5px' height="80px"></img>
#### Resnet152 <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query1-women_dress/query1-res.jpg' padding='5px' height="80px"></img>

### query2 - orange
#### query <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query2-orange/query2.jpg' padding='5px' height="80px"></img>
#### color <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query2-orange/query2-color.jpg' padding='5px' height="80px"></img>
#### daisy <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query2-orange/query2-daisy.jpg' padding='5px' height="80px"></img>
#### edge <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query2-orange/query2-edge.jpg' padding='5px' height="80px"></img>
#### gabor <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query2-orange/query2-gabor.jpg' padding='5px' height="80px"></img>
#### HOG <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query2-orange/query2-hog.jpg' padding='5px' height="80px"></img>
#### VGG19 <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query2-orange/query2-vgg.jpg' padding='5px' height="80px"></img>
#### Resnet152 <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query2-orange/query2-res.jpg' padding='5px' height="80px"></img>

### query3 - NBA jersey
#### query <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query3-nba_jersey/query3.jpg' padding='5px' height="80px"></img>
#### color <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query3-nba_jersey/query3-color.jpg' padding='5px' height="80px"></img>
#### daisy <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query3-nba_jersey/query3-daisy.jpg' padding='5px' height="80px"></img>
#### edge <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query3-nba_jersey/query3-edge.jpg' padding='5px' height="80px"></img>
#### gabor <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query3-nba_jersey/query3-gabor.jpg' padding='5px' height="80px"></img>
#### HOG <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query3-nba_jersey/query3-hog.jpg' padding='5px' height="80px"></img>
#### VGG19 <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query3-nba_jersey/query3-vgg.jpg' padding='5px' height="80px"></img>
#### Resnet152 <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query3-nba_jersey/query3-res.jpg' padding='5px' height="80px"></img>

### query4 - snack
#### query <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query4-snack/query4.jpg' padding='5px' height="80px"></img>
#### color <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query4-snack/query4-color.jpg' padding='5px' height="80px"></img>
#### daisy <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query4-snack/query4-daisy.jpg' padding='5px' height="80px"></img>
#### edge <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query4-snack/query4-edge.jpg' padding='5px' height="80px"></img>
#### gabor <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query4-snack/query4-gabor.jpg' padding='5px' height="80px"></img>
#### HOG <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query4-snack/query4-hog.jpg' padding='5px' height="80px"></img>
#### VGG19 <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query4-snack/query4-vgg.jpg' padding='5px' height="80px"></img>
#### Resnet152 <img align='center' style="border-color:gray;border-width:2px;border-style:dashed" src='result/retrieval_result/query4-snack/query4-res.jpg' padding='5px' height="80px"></img>



## 4. 项目的使用
如果你对结果感兴趣，并想要尝试自己的图像，

请参阅 [USAGE.md](https://github.com/pochih/CBIR/blob/master/USAGE.md), [USAGE_zh.md](./USAGE_zh.md)

细节写在这里面



## 作者
Po-Chih Huang / [@pochih](http://pochih.github.io/)
### 翻译与扩充
Dedicatus1979 / [@Dedicatus1979](https://github.com/Dedicatus1979/)

## -1. 译者后话
我为本项目添加了requirement.txt 可以确保在使用这些包的情况下在python3.10版本下可以使用

然后，我把其中一些版本过时的用法给去除了，例如将scipy.misc.imread()，给替换成了Image.open().convert('RGB')，然后再np.array()

