## 首先检查loss on training data,然后再检查loss on testing datas

============================================

## loss on training data Large

### 1.Model Bias

- Model太简单,重新设计Model,给更大的弹性,增加输入的feature/用Deep Learning(more neurons,layers)
### 2.Optimization做的不好

- 这门课只用Gradient Descent.可能找到的不是Global minima

## 怎么看是上面哪种情况呢 ?
## solution：比较不同的模型来看Model够不够大。
## 层数越多弹性越大就越不是Model Bias,排除法能看出来是不是Optimization
## 线性层或者支持向量机这种简单的Model的Optimization是没有问题的.然后再试一下Deeper networks

## loss on training data small

============================================

## loss on testing data 
## loss on testing data 小:结束了

============================================

## loss on testing data 大:

## training loss小,testing loss 大.这种情况是over fitting.有弹性的Model容易overfitting.

## 解决办法:

- 1.增加训练资料,最简单的方法.
- 2.Data augmentation.自己创造出资料. e.g.图像识别中可以左右翻转...  但是不能随便Data augmentation,一定要有道理,比如不能上下颠倒.一定要合理!
- 3.constrained model,减小Model的弹性.Model的设计取决于对问题的理解.Less parameters,sharing parameters.之前讲的架构叫Fully-connected(全连接),有弹性.CNN相对而言没有太大的弹性,但是专门针对影像处理. Less features,Early stopping Regularization Dropout...

# **一定要先看training loss**

## public Testing Set 和 private Testing Set都要好,benchmark corpora(public)上面表现的好还要应用于private Testing Set.

# 把Training Set分成Training Set 和Validation Set
## 如果怕分不好,就用N-fold Cross Validation,然后排列组合一下就可以又更好的评估方式

# mismatch的原因和overfitting不一样.
# mismatch:training data和testing data的distributuions不一样.
