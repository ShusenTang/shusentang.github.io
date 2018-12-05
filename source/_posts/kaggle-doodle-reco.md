---
title: Quick, Draw! Doodle Recognition Challenge 总结
date: 2018-12-05 21:58:29
toc: true
mathjax: true
categories: 
- Competitions
tags:
- kaggle
- classification
---

<center>
<img src="./kaggle-doodle-reco/cover.png" width="900" class="full-image">
</center>

这个比赛于今早结束，最终结果是排在69/1316、铜牌，离银牌区差4名，还是比较遗憾的。不过这是我第一次花大量时间和精力在图像分类问题上，能拿到牌还算满意。

这篇博客主要记录我在这次比赛过程中学到的一些东西，由于是第一次花精力认真做图像类的比赛所以学到的东西还是很多的。最后再会总结一下排在前列的大佬们在讨论区分享的他们的方案，留作日后参考。

<!-- more -->

# 1. 赛题简述
## 1.1 数据集
还记得前段时间很火的微信小程序“猜画小歌”吗，[这个比赛](https://www.kaggle.com/c/quickdraw-doodle-recognition)的数据就来自这个小程序的网页版[Quick, Draw!](https://quickdraw.withgoogle.com/)，游戏提示用户绘制描绘特定类别的涂鸦，例如“香蕉”，“桌子”等，所以Google通过这个小游戏收集了来自世界各地的涂鸦数据, 数据集所有信息都可见此数据集的[官方仓库](https://github.com/googlecreativelab/quickdraw-dataset#the-raw-moderated-dataset)。本次比赛使用了340类一共约五千万个样本。

主办方给了两个版本的训练集，raw 和 simpled，都是以csv文件的形式给出的。raw版本就是原始收集到的数据各字段如下所示:

| Key          | Type                   | Description                                  |
| ------------ | -----------------------| -------------------------------------------- |
| key_id       | 64-bit unsigned integer| 独一无二的样本ID     |
| word         | string                 | 样本所属类别    |
| recognized   | bool               | 样本是否被游戏识别 |
| timestamp    | datetime             | 样本创建时间                |
| countrycode  | string                 | player所在国家代码([ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2))|
| drawing      | string                 | 涂鸦数据信息 |  

其中drawing字段就是涂鸦数据，包含坐标和时间信息，示例如下:
```
[ 
  [  // First stroke 
    [x0, x1, x2, x3, ...],
    [y0, y1, y2, y3, ...],
    [t0, t1, t2, t3, ...]
  ],
  [  // Second stroke
    [x0, x1, x2, x3, ...],
    [y0, y1, y2, y3, ...],
    [t0, t1, t2, t3, ...]
  ],
  ... // Additional strokes
]
```

raw版本的数据集很大而且很多冗余信息，所以大多数选手(包括我)都是用的simple版本作为主要训练集，simpled数据集去掉了时间信息和冗余的坐标信息(比如两点确定一条线段，那么线段中间的点就是冗余的)并将坐标进行了scale，具体处理方式如下:   
* scaled 的坐标数据进行了左上对齐，最小值为0最大值为255.
* 进行了重采样使坐标都是0-255的整数.
* 去除冗余坐标使用的是Ramer–Douglas–Peucker算法，epsilon设为2.0；

更多信息可见此数据集的[官方仓库](https://github.com/googlecreativelab/quickdraw-dataset#the-raw-moderated-dataset)。

## 1.2 赛题任务
本题的任务就是预测测试集涂鸦属于哪个类别，是一个单分类问题。此题的难度在于，由于训练数据来自游戏本身，涂鸦样本可能不完整而且可能与标签不符，噪声比较多，如图1.1所示([图片来源](https://www.kaggle.com/gaborfodor/un-recognized-drawings/output))。选手需要构建一个识别器，可以有效地从这些噪声数据中学习得到模型。
<center>
    <img src="./kaggle-doodle-reco/1.1.png" width="500" class="full-image">
    <br>
    <div style="border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;">图1.1</div>
</center>

上图就是训练集中属于“蚊子”类别的数据示例，绿色是被标记为"可识别"的样本，红色是被标记为"不可识别"的样本，可以看到不管是可识别还是不可识别的样本，都存在噪声的情况。

## 1.3 评价指标



# 2. 方案
可见此题的数据是序列数据，所以最先想到可以用RNN，当然把数据渲染成图片也可以用CNN。


# 3. top方案总结


