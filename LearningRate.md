# Adaptive Learning Rate

---

loss不在下降无法说明gradient很小.

目前用gradient descent 做optimization一般的问题都不是critical point

training卡住的原因:

并不是Learning Rate太大.

## 客制化Learning Rate

平坦Larger 陡峭Smaller

- 1.Root Mean Square   用在Adagrade

gradient大,Learning Rate就小

考虑到过去的全部的gradient

- 2.RMSProp (找不到论文)

第一步和Root Mean Square一样
第二步的时候要加权(现在的gradient有多重要)

## ==最常用的Optimization的策略Optimizer:  Adam:RMSProp + Momentum==
Original paper: https://arxiv.org/pdf/1412.6980.pdf
pytorch里面都写好了，预设的参数是比较好的

## Learning Rate Scheduling

上面的η要和时间有关,而不是常数.

- Learning Rate Decay
随着时间增加,Learning Rate逐渐减小。因为越接近终点,他的学习速率就要慢慢变小。
这种方法很好地解决了Root Mean Square "乱喷" 现象.

- Warm Up
Learning Rate随着时间先变大后变小.
在训练Bert的时候往往会用到,但是他在原始时代就有了. 
e.g.Residual Network里面有Warm Up 
> https://arxiv.org/abs/1512.03385 

 15年12月放在了arxiv上.(上古时代) 先用0.01再用0.1  最常见的都是越来越小,这里反其道而行之.
在Transformer里面也有提到.
> https://arxiv.org/abs/1706.03762

解释理由:刚开始先探索,统计一些数据再让Learning Rate变大.

RAdam是Adam的进阶

现在对Adam的改进都是去计算momentum η σ
