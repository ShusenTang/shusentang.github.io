---
title: TensorFlow Attention源码终极解析(超详细图解)
date: 2019-03-09 18:37:19
toc: true
mathjax: true
categories: 
- Deep Learning
tags:
- tensorflow
- attention
- seq2seq
---

<center>
<img src="./tf-attention/cover.png" width="500" class="full-image">
</center>

Attention在seq2seq模型中是一个很有用的机制，由于TensorFlow烂成翔的官方文档以及网上很少而且晦涩难懂的教程，我在如何正确使用TensorFlow现成attention接口上面费了很大一番功夫。本文用详细图解的方式清晰展现了其源代码构成，方便大家学习使用。本文会简略的介绍一下seq2seq attention的原理，然后详细剖析TensorFlow相关的源代码。

<!-- more -->

# 1. 原理简介
## 1.1 seq2seq

seq2seq是"Sequence to Sequence"的简写，seq2seq模型的核心就是编码器（Encoder）和解码器（Decoder）组成的，该架构相继在论文[1][2]提出，后来[3]又提出了在seq2seq结构中加入Attention机制，使seq2seq的性能大大提升，现在seq2seq被广泛地用于机器翻译、对话生成、人体姿态序列生成等各种任务上，并取得了非常好的效果。

<center>
    <img src="./tf-attention/1.1.png" width="500" class="full-image">
    <br>
    <div style="border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;">图1.1 seq2seq基本结构</div>
</center>

未加入attention机制的seq2seq基本模型如上图所示，左侧的RNN编码器将输入编码成一个中间状态c向量，右侧的解码器在每一个时间步都会接收c的输入然后进行解码，得到输出结果。例如如果编码器的输入为“我爱你”，编码器的期望输出为“I love you”，那么seq2seq就实现了翻译任务，以上就是最基本的seq2seq模型原理。

## 1.2 attention
仔细观察图1.1所示的模型就可以发现一个问题，c向量是连接编码器和解码器唯一的桥梁，那么c向量中必须包含了原始序列中的所有信息，当编码器输入序列很长的时候，由于RNN并不能很好地对很长的序列进行建模，c向量就很难囊括输入序列所有信息了，此时解码器也就很难正确地生成了。

Attention机制解决了这个问题，它可以使得在输入序列长的时候精确率也不会有明显下降，它是怎么做的呢？既然一个c向量存不了，那么就引入多个c向量: $c_1, c_2, ...$，称为context vector。

<center>
    <img src="./tf-attention/1.2.png" width="500" class="full-image">
    <br>
    <div style="border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;">图1.2 带attention的seq2seq结构</div>
</center>

带attention的seq2seq结构如图1.2所示，attention机制跟人类翻译句子时候的思路有些类似，即将注意力关注于我们翻译词语对应的上下文（context）。而 $c_i$ 就存放了此刻输入的上下文信息，这样当解码器解码到第 $i$ 步的时候通过分析此时输入序列的上下文信息即可精准地进行解码。

下面用公式化的语言叙述一下加入attention与否的seq2seq结构。
在没有引入attention之前，编码器在某个时刻解码的时候实际上是依赖于三个部分的（图1.1没有画出），首先我们知道RNN中，每次输出结果会依赖于隐藏态和输入（在训练过程中这个输入就是ground truth，在测试时这个输入就是上一步的输出），在seq2seq模型中，还需要依赖于c向量，所以这里我们设在 $i$ 时刻，解码器解码的期望输出是 $y_i$，上一次解码结果（的ground truth）是 $y_{i-1}$，此时刻隐藏态是 $h_i'$，所以它们满足这样的关系：
$$
y_i = f(y_{i-1}, h_i', c) \tag{1.1}
$$
即每次的解码输出是上一个隐藏态和上一个输出结果和c向量共同计算得出的。

那么加入attention机制后，将每时刻固定的c向量换成 $c_i$，有
$$
y_i = f(y_{i-1}, h_i', c_i) \tag{1.2}
$$

那么上下文信息 $c_i$ 是怎么计算得来的呢？目前常用的两种方式就是Bahdanau attention[3]和Luong attention[4]，下面就分别来介绍一下。

### 1.2.1 Bahdanau attention

假设编码器的所有时刻的隐藏态（hidden state）为 $h_1, h_2, ..., h_{T_x}$，则有
$$
c_i = \sum_{j = 1}^{T_x} \alpha_{ij}h_j \tag{1.3}
$$
这里 $\alpha_{ij}$ 为每一个编码器输出的隐藏态 $h_j$ 被分配到的权重，它的值越大代表对编码器 $j$ 时刻输入的注意力就越大。而 $\alpha_{ij}$ 的计算法方式如下：
$$
\alpha _ { i j } = \frac { \exp \left( e _ { i j } \right) } { \sum _ { k = 1 } ^ { T _ { x } } \exp \left( e _ { i k } \right) } \tag{1.4}
$$
$$
e _ { i j } = a \left( h_{i-1}' , h _ { j } \right) = v _ { a } ^ { T } \tanh \left( W _ { a } h_{i-1}' + U _ { a } h _ { j } \right) \tag{1.5}
$$
也就是说，权重 $\alpha_{ij}$ 就是解码器隐藏态 $h_{i-1}'$ 和编码器隐藏态 $h_j$ 计算得到一个数值 $e_{ij}$ （一般称为得分，score）再经过softmax归一化得到的。

### 1.2.2 Luong attention

### 1.2.3 小节
综上，attention机制的大致原理就是通过在解码的每一步计算一下此刻应该对编码器每时刻的输入赋予多少注意力，然后再有针对性地进行解码。但除了计算分数的方式有很多方式，attention的具体细节花样还有很多，例如是否将解码器前一步得到的 $c_{i-1}$ 作为此刻解码器的输入，是否将编码器所有隐藏态都参与注意力计算等等。下一节我们通过对TensorFlow的源码进行剖析来看看TensorFlow是怎么实现的。


# 2. TensorFlow源码剖析

## 2.1 概览
在TensorFlow中，Attention 的相关实现代码是在[tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py](https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py)文件中，这里面我们需要关注的有如下几个类：
* `AttentionMechanism`: 所有attention机制的父类，内部没有任何实现。
* `_BaseAttentionMechanism`: 继承自`AttentionMechanism`，定义了attention机制的一些公共方法实现和属性。
* `BahdanauAttention`和`LuongAttention`：均继承自`_BaseAttentionMechanism`，分别实现了1.2节所述的两种attention机制。
* `AttentionWrapper`: 用于封装RNNCell的类，继承自`RNNCell`，所以被它封装后依然是一个RNNCell类，只不过是带了attention的功能。
* `AttentionWrapperState`：用来存放计算过程中的state，前面说了`AttentionWrapper`其实也是一个RNNCell，那么它也有隐藏态（hidden state）信息，`AttentionWrapperState`就是这个state。除了RNN cell state，其中还额外存放了一些信息。

此外还有`_BaseMonotonicAttentionMechanism`、`BahdanauMonotonicAttention`、`LuongMonotonicAttention`以实现单调（monotonic）attention机制，以及一些公共的方法，如`_luong_score`、`_bahdanau_score`、`_prepare_memory`等等。

在进一步分析之前，我们先来明确代码中一些术语的意思：
* key & query: Attention的本质可以被描述为一个查询（query）与一系列（键key-值value）对一起映射成一个输出：将query和每个key进行相似度计算得到权重并进行归一化，将权重和相应的键值value进行加权求和得到最后的attention，这里key=value。简单理解就是，query相当于前面说的解码器的隐藏态 $h_i'$ ，而key就是编码器的隐藏态 $h_i$。
* memory: 这个memory其实才是编码器的所有隐藏态，与前面的key区别就是key可能是memory经过处理（例如线性变换）后得到的。

## 2.2 `_BaseAttentionMechanism`
先来看看最基础的attention类`_BaseAttentionMechanism`。它的初始化方法如下：
``` python
def __init__(self,
            query_layer,
            memory,
            probability_fn,
            memory_sequence_length=None,
            memory_layer=None,
            check_inner_dims_defined=True,
            score_mask_value=None,
            name=None):
```
这里有很多参数，下面一一说明这些参数的作用：
* `query_layer`: 一个`tf.layers.Layer`实例，query会首先经过这一层；
* `memory`: 解码时用到的所有上下文信息，可简单理解为编码器的所有隐藏态，维度一般为[batch, max_time, enc_rnn_size]；
* `probability_fn`: 将score $e_{ij}$ 计算成概率用的函数，默认使用softmax，还可以指定hardmax等函数；
* `memory_sequence_length`: 即memory变量的实际长度信息，类似`dynamic_rnn`中的`sequence_length`，维度为[batch]，这会被用作mask来去除超过实际长度的无用信息；
* `memory_layer`: 类似`query_layer`，也是一个`tf.layers.Layer`实例（或None），memory会经过这一层然后得到keys。需要注意的是，**（经过`memory_layer`处理后得到的）key应该和（经过`query_layer`处理得到的）query的维度相匹配**，用式$(1.5)$为例就是 $W_a h_{i-1}'$ 要和 $U_ah_j$ 的维度要一致，不然没法相加。
* `check_inner_dims_defined`: bool型，是否检查memory除了最外面两维其他维度是否是有定义的。
* `score_mask_value`: 在使用`probability_fn`计算概率之前，对`score`预先进行mask使用的值，默认是负无穷。但这个只有在`memory_sequence_length`参数定义的时候有效。

其构造函数完成的主要功能就是将输入的memory进行mask然后经过`memory_layer`处理称为`keys`方便后面使用，如下面代码所示：
``` python
self._values = _prepare_memory(memory, memory_sequence_length,
          check_inner_dims_defined=check_inner_dims_defined)
self._keys = (self.memory_layer(self._values) if self.memory_layer  else self._values)
```

## 2.3 `BahdanauAttention`
`BahdanauAttention`是继承自2.2节的`_BaseAttentionMechanism`，它的构造函数如下所示：
``` python
def __init__(self,
            num_units,
            memory,
            memory_sequence_length=None,
            normalize=False,
            probability_fn=None,
            score_mask_value=None,
            dtype=None,
            name="BahdanauAttention"):
```
这些参数中，除了`num_units`其他都在`_BaseAttentionMechanism`出现过这里不再赘述。主要说一下`num_units`，我们知道在计算式$(1.5)$的时候，需要使用 $h_{i-1}'$ 和 $h_i$ 来进行计算，而二者的维度可能并不是统一的，需要进行变换和统一，所以就有了 $W_a$ 和 $U_a$ 这两个系数，所以在代码中就是用`num_units`来声明了一个全连接 Dense 网络，用于统一二者的维度：
``` python
super(BahdanauAttention, self).__init__(
        query_layer=layers_core.Dense(num_units, name="query_layer", use_bias=False, dtype=dtype),
        memory_layer=layers_core.Dense(num_units, name="memory_layer", use_bias=False, dtype=dtype),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
```
即这里的`num_units`确定了需要传进`_BaseAttentionMechanism`构造函数的参数`query_layer`和`memory_layer`。


# 参考文献
[1] Cho K, Van Merriënboer B, Gulcehre C, et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation[J]. arXiv preprint arXiv:1406.1078, 2014.

[2] Sutskever I, Vinyals O, Le Q V. Sequence to sequence learning with neural networks[C]//Advances in neural information processing systems. 2014: 3104-3112.

[3] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[J]. arXiv preprint arXiv:1409.0473, 2014.

[4] Luong M T, Pham H, Manning C D. Effective approaches to attention-based neural machine translation[J]. arXiv preprint arXiv:1508.04025, 2015.



