# Transformer
## Sequence-to-sequence(Seq2seq)
## 输入和输出的长度由机器决定
#### 语音识别，机器翻译，语音翻译
# *硬train一发！*
## Seq2Seq一般分为Encoder和Decoder
## Transformer里面的Encoder用的Self-Attention
## Residual connection:把input和output相加
## 每一个block里面有好多layer
## layer normalization:计算mean和deviation，不用考虑batchsize
# 建议看图理解过程
# BERT就是transformer的Encoder

# Self-Attention架构
## 计算vector里面每个维度之间的relevant，
