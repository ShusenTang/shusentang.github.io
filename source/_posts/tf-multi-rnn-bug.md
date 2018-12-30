---
title: TensorFlow实现多层RNN的一个大坑
date: 2018-11-13 22:30:02
toc: true
mathjax: true
categories: 
- Deep Learning
tags:
- tensorflow
- RNN
- attention
- seq2seq
---

<center>
<img src="./tf-multi-rnn-bug/cover.jpg" width="400" class="full-image">
</center>

<!-- more -->

## 起因

事情的起因是这样的，我已经用tensorflow实现了一个带attention的encoder-decoder(都是单层的RNN)的结构，代码组织结构如下所示
``` python
encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_size)
decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_size)

def Encoder(cell, inputs):
    '''根据输入得到输出'''
    ......
    return outputs

# shape: (batch_size, max_seq_len, rnn_size)
encoder_outputs = Encoder(encoder_cell, inputs)

# 下面是attention
attn_mech = tf.contrib.seq2seq.LuongAttention(...) 
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attn_mech, attention_layer_size=attn_size,...)

# 下面的就不重要了
......
```
上面这段代码在attn_size为任何值的时候都是可以正常执行的。这也很符合预期，因为上面这段代码所干的事情如下:
* 用encoder将input编码成encoder_output(即attention的keys或者memory)；
* 对于decode的每一个时刻t，将t-1时刻得到的attention context(shape[-1]为attn_size)和decoder的输入合并在一起输入到decoder；
......

可以看到attn_size确实是任何值都可以, 也即decoder的输入维度(attn_size + input_x_size)为任何都可以。
> 注意TensorFlow中的RNN cell不需要显式指定输入的维度(而是自己推断出来)，这和pytorch不一样:
`pytorch_rnn = torch.nn.LSTM(input_size = attn_size + input_x_size, hidden_size=rnn_size)`

## 经过 
后来我又想将decoder改成多层的RNN，decoder结构就像下面右边这样：
<center>
<img src="./tf-multi-rnn-bug/enc_dec_with_attn.png" width="400" class="full-image">
</center>

于是我将`decoder_cell`的定义做了如下修改:
``` python
......

one_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_size)
decoder_cell = tf.nn.rnn_cell.MultiRNNCell([one_cell for _ in range(dec_num_layers)])
......
```
除非把`attn_size`设置成`rnn_size - input_x_size`，否则会报类似下面的维度不对的错误(假设`rnn_size=256`, `attn_size + input_x_size = 356`)
```
ValueError: Dimensions must be equal, but are 256 and 356 for 'rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/lstm_cell/MatMul_1' (op: 'MatMul') with input shapes: [30,256], [356,1200].
```

这是为什么呢？明明按照前面的分析，明明attn_size设置成任何值都可以的啊。

## 解决
一开始我一直以为是我的attention写得不对，于是google了好久都没发现attention问题在哪？
直到我看到了这个[issue](https://github.com/tensorflow/tensorflow/issues/16186)才发现是我的多层RNN没写对，还是自己太菜了😭

正确的多层`decoder_cell`应该是如下定义:
``` python
......
cell_list = [tf.nn.rnn_cell.LSTMCell(num_units=rnn_size) for _ in range(dec_num_layers)]
decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
......
```

咋一看上面这段代码貌似和之前的错误代码没什么区别，但是如下代码你就应该意识到哪儿不对了
``` python
>>> str = "bug"
>>> strs = [str for _ in range(2)]
>>> print(strs)
['bug', 'bug']
>>> for str in strs:
        print(id(str)) # id()函数用于获取对象的内存地址
4367049200
4367049200
```
注意到上面输出的两个地址都是一样的。因此，我们就知道问题出在哪儿了:  
对于前面错误的多层rnn实现, 每一层的LSTMCell其实都是同一个(指向它们的指针是相同的)，那么每一层的LSTMCell的weights维度就也是一样的，但其实第一层的输入维度(`attn_size + input_x_size`)和其它层的（`rnn_size`)一般都是不一样的，如下图所示，这样就会报维度错误了。
<center>
<img src="./tf-multi-rnn-bug/enc_dec_with_attn_2.png" width="400" class="full-image">
</center>

而正确代码中，每一个LSTMCell都是通过`tf.nn.rnn_cell.LSTMCell(num_units=rnn_size)`定义的，因此可以有不同的结构，自然不会报错。

## 总结

* TensorFlow中错误的多层RNN实现方式:
``` python
one_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_size)
decoder_cell = tf.nn.rnn_cell.MultiRNNCell([one_cell for _ in range(dec_num_layers)])
# decoder_cell = tf.nn.rnn_cell.MultiRNNCell([one_cell]*dec_num_layers])也是错误的
```

* TensorFlow中正确的多层RNN实现方式:
``` python
cell_list = [tf.nn.rnn_cell.LSTMCell(num_units=rnn_size) for _ in range(dec_num_layers)]
decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
```


参考
1. [Cannot stack LSTM with MultiRNNCell and dynamic_rnn](https://stackoverflow.com/questions/47371608/cannot-stack-lstm-with-multirnncell-and-dynamic-rnn)
2. [using dynamic_rnn with multiRNN gives error](https://stackoverflow.com/questions/48865554/using-dynamic-rnn-with-multirnn-gives-error/53277463#53277463)