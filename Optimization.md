# 怎么把gradient descent做得更好 ?  

## gradient is close to zero : local minima ?

### 实则不是! gradient = 0 的地方还有saddle point(马鞍面).这些都是critical point.

有办法看是local minima 还是saddle point
### 为什么要研究这个问题? 因为local minima没有地方走,而saddle point还可以下降!
---

### 证明:微积分＋线性代数  要去算loss function  
#### Tayler Series Approximation在θ'附近泰勒展开
$ L(θ) ≈ L(θ') + (θ - θ')T g + 1/2(θ - θ')T H (θ - θ') $

gradient g is a vector 一阶偏导
Hessian H is a matrix 二阶偏导

critical point的情况只剩下Hessian,第二项为0
$令v = θ - θ'$  
任意的 vTHv > 0 则L( θ ) > L( θ' ) 说明是Local minima
任意的 vTHv < 0 则L( θ ) < L( θ' ) 说明是Local maxima
$vTHv$ 时正时负, 则 Saddle point

看所有的v不现实,则用线性代数的结论.看H是不是正定矩阵/负定矩阵/不定矩阵,也就是算所有eigen values(特征值)的符号

H may tell us parameter update direction!

u is an eigen vector of H
λ is an eigen value of u 
$uTHu = uT(λu) = λu²$  
λ < 0 则 L( θ ) < L( θ' ).沿着u的方向就可以让loss变小
也就是找出负的eigen value和它对应的eigen vector

---

### 但是实际上这样运算量太大了,几乎没有人这么去做来逃离Saddle point.讲这个是为了说明我们遇到Saddle point是有机会继续往哪一个方向走的!

Saddle Point v.s. Local Minima
如果升维是会产生新路. Local Minima在更高的维度来说可能是Saddle Point.
#### 一般情况下Saddle Point 要比Local Minima多很多.极端情况下也就才五五开.

---

# Batch and Momentum  

## Batch  
实际上算微分的时候不是对所有的L做微分，而是分成一个一个小的batch.  
所有的batch都看过后叫一个epoch  
Shuffle常见做法:Shuffle after each epoch
为什么用Batch?  
Batch size = N(Full batch)  看过20个examples后才Update一次
Batch size = 1 每次看一个example就可以Update一次,但是noisy
因为有GPU所以1和1000是一样的时间,但是1和60000确实会增加一些时间 (Tesla V100GPU)GPU平行运算.
所以大的Batch 和小的Batch的时间是差不多的.
大的Batch比较稳定,小的Batch比较noisy.
但是noisy gradient反而容易training
Batchsize越大往往在training情况下有较差的结果.Optimization Fails.
*Smaller batch size has better performance*
*"Noisy" update is better for training*
Minima里面可分好的minima和坏的minima

- Flat Minima(small batch)
- Sharp Minima(large batch)

所以大的batch容易overfitting

鱼和熊掌能否兼得?

有很多paper,76分钟train BERT.50分钟train ResNet...

**MNIST**:手写数字图片的数据集
**CIFAR-10**:影像辨识

## Momentum
一种可以对抗Saddle Point 和 Local Minima的技术
从物理学中加入了惯性.

## Gradient Descent + Momentum(动量)  
gradient 加上 前一步的方向(向量相加),这两个向量前面都要加入系数,系数也是自己调整的
