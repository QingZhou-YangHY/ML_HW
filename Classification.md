# Classification

---

## Regression

输入一个向量,输出一个数值,和目标label越接近越好

Classification as regression?

输入一个x,输出一个y,要和正确的class越接近越好.

这个方法有时候行,有时候不行.


## Classification

## Class as one-hot vector

把本来output一个数值变成重复输出若干个数值.这样就是一个向量了.

- ==往往在最后要加入softmax==
  - 原因:把y里面的任何值变成0-1之间的数.  
$y' = e^{y_{i}}/\sum_{j}e^{y_{j}}$


- Sigmod本质上是Softmax的特殊情况.
  - Sigmod:二分类,多标签
  - softmax:多分类,概率和为1

## Loss of Classification

- Mean Square Error(MSE)
  - $e~=~\sum_{i}(\hat{y_{i}}-y_{i}')^{2}$

- ==Cross-entropy==
  - $e~=~-\sum_{i}\hat{y_{i}}lny_{i}'$

Minimizing cross-entropy is equivalent to maximizing likelihood.一模一样,不同的讲法而已.

pytorch中Cross-entropy和softmax是绑定在一起的.Cross-entropy里面有softmax.不用自己加了.

选择Cross-entropy的原因:
数学证明有链接.
下面举个例子:
$\quad $ MSE再large loss的地方没有斜率,卡住了.而Cross-entropy没有这个问题.所以Classification一般都选Cross-entropy.不用去使劲调Optimization了.
Loss function的定义可能会影响training是否容易.改变Loss function来改变Optimization的难度.
