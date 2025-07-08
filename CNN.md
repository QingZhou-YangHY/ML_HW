# CNN(Convolutional Neural Network)

- Network Architecture designed for Image

---
## Image Classification

==模型的参数越多,弹性越大,越容易overfitting==

一张图片是一个三维的tensor.每一个pixel都是由RGB组成的.3个channel代表了RGB三个颜色.长和宽代表了解析度,像素的数量.
<br/>

- Observation 1
  - A neuron does not have to see the whole image.
<br/>

Neuron Version Story
- Simplification 1
  - 只让他看一小部分

  - Receptive field:拉直作为neuron的输入,然后输出作为下一层的输入.每一层的neuron只考虑自己的Receptive field.Receptive field可以重叠.多个neuron可以处理同一个Receptive field.
  -Receptive field是根据自己的理解设计的,没有太多限制.通常都是相邻的领地.

  - Typical Setting最经典的,会看所有的channel.所以描述的时候只考虑高和宽,叫kernel size.不会设太大,影像辨识3 x 3就已经够了. 7 x 7 , 9 x 9蛮大的kernel size.
  
  - 一般来说一个receptive field 会有一组neuron.例如:64/128个来处理.


  - 一步的量叫stride,往往设1或2,不会设太大,希望Receptive field有重叠,不要miss掉pattern.

  - 超出了影像的范围的解决办法:padding(补0).也有其他补值的方法,平均或者边缘的数据.
<br/>

- Observation 2
  - The same patterns appear in different regions.
  - Each receptive field needs a "break" detector?
<br/>

- Simlification 2
  - 让不同receptive field 共享参数.
  - Typical Setting
  Each receptive field has a set of neurons(e.g.,64neurons)
  - 每一个Receptive field都只有一组参数,这些参数的名字叫filter(两个neuron共用的同一组参数).一个Receptive field里面有filter 1 , filter 2, filter 3...


Fully Connected Layer只看了一个小范围变成了Receptive Field.弹性变小.参数共享进一步限制了弹性.
<br/>
==Receptive Field + Parameter Sharing 就是 Convolutional Layer==
用Convolutional Layer的Network叫Convolutional Neural Network(CNN)

Convolutional Layer的第二种讲解方式（Filter Version Story）就是很常见的卷积一遍得到一个新的Feature Map.(每一个channel代表一个filter)

Convolutional Layer 有 Larger model bias(for image),不容易overfitting.

Filter扫过一张图片叫Convolution.
<br/>

- Observation 3
  - Subsampling the pixels will not change the object
<br/>

- Simlification 3(可有可无)

  - Pooling(池化):做Subsampling(二次采样) ,课上讲的是Max Pooling
  - Max Pooling的方法:每一组里面选一个最大的值作为代表.
  - 没有参数,没有weight这些要学的东西
<br/>

做完Convolution后面会接一个Pooling.Convolution和Pooling交替使用.比如两次Convolution后面接一次Pooling.

但是Pooling对于performance会带来伤害.比如侦测的是非常细微的东西.这样Subsampling,performance会差一点.近年来很多影像辨识Pooling逐渐被抛弃,做Full Convolutional Network.因为Pooling主要是为了减少运算量.

Flatten:把矩阵拉直变成向量.

下围棋就是分类问题. 19 x 19 classes  48个channel
Fully-connected network can be used.
But CNN performs much better.
Normalization.

==CNN是专门用于影像上面的.==
因为影像和Go playing有很多相似之处.
Alpha Go的paper里面附件说了network.19 x 19 x 48 ，zero pads ，23 x 23 image ，k filters of kernel size 5 x 5 , k = 192 filters,stride 1 , rectifier nonlinearity...
Alpha Go does not use Pooling...
下围棋就是不适合用Pooling!

CNN不能够处理影像放大缩小旋转等等,所以要data augmentation
