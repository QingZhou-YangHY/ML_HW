# Transformer
Sequence-to-sequence(Seq2seq)
输入和输出的长度由机器决定
语音识别，机器翻译，语音翻译
# *硬train一发！*
Seq2Seq一般分为Encoder和Decoder
Transformer里面的Encoder用的Self-Attention
Residual connection:把input和output相加
每一个block里面有好多layer
layer normalization:计算mean和deviation，不用考虑batchsize
建议看图理解过程
BERT就是transformer的Encoder

Self-Attention架构
计算vector里面每个维度之间的relevant，

# Self-Attention(自注意力机制)
一种方法:One-hot Encoding 问题:无法反映语义上面的关系
另一种方法:Word Embedding 
POS tagging:词性标注

不是考虑一个window,而是考虑the whole sequence
用到Self-attention吃一整个sequence,输入几个vector就output几个vector(考虑到整个句子),可以叠加很多次.FC和Self-attention可以交替使用. Attention is all you need.这篇paper提出了transformer架构.

举个例子.根据a1向量找出sequence里面与a1相关的几个向量.每一个向量和a1的关联程度用α表示.
计算attention的模组(如何计算α): Dot-product 把输入的两个向量乘上不同的矩阵,得到q和k,做内积得到α.  Additive：把输入的两个向量乘上不同的矩阵,q和k加起来丢入tanh,通过transform得到α. 下面只讨论Dot-product,也是今天最常用的方法,用在transformer中.

接下来计算a1和其他所有的关联性α.query1和key1 key2内积得到attention score α1,1 α1,2 ...   接下来Soft-max和分类时候的Soft-max是一样的.得到一排新的α.(其实这里面不一定要用Soft-max,随便一种Activation)   接下来用a1 a2乘Wv得到v1 v2 ... 把v1 v2 ...乘上对应的α再相加b1

每一个a都产生q k v          qi = Wq * ai    ki = Wk * ai   vi = Wv * ai  

只有Wq Wk Wv是未知的,需要学习的.通过training data 找出来.

进阶:Multil-head Self-attention 几种不同的相关性就几个head 几个q k v(把q k v 分别乘上两个矩阵)

上面少了位置的资讯No position information in self-attention

可以加入Positional Encoding,对每一个position 都有一个独特的vector ei,把ei加入到ai上就结束了。但这个方法是hand-crafted

Self-attention 应用于Transformer 在nlp(Bert原来就是个做填空题的hhh),speech上面也有用到,影像上也有用,一张图片可以看成是vector set

Truncated Self-attention在语音辨识上面只考虑上下文一小个范围

很多的应用都把RNN换成了Self-attention. Self-attention WIN! Self-attention for Graph 就是某一种GNN

Self-attention的变形就是xxformer
