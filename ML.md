# Machine Learning ≈ Looking for Function
=========================================
## Different types of Functions 两大类问题Regression Classification
- Regression:The function outputs a scalar(数值).
- Classification:Given options(classes),the function outputs the correct one.不一定只有两个选项,可以有多个选项。

    e.g.Playing GO本质上也是一个选择题(19 x 19)

- Structured Learning:create something with structure(image,document)

## **机器学习的三个步骤**

### **1. Function with Unknown Parameters**

based on domain knowledge 来写出函数表达式(猜测)
带有未知参数的funtion叫Model.已知的x1叫feature. w叫weight,b叫bias.($y = b + wx1$)

### **2. Define Loss from Training Data**

Loss也是一个function,输入是Model里面的参数,输出代表设置参数的好坏.Loss要从训练的资料进行计算.算已知的然后看拟合的好不好.
正确/真实的值叫label.
有很多种办法算Loss. 
例:
- MAE:Mean absolute error
- MSE: MEan square error
- Cross-entropy:y和y head 都是几率的分布(适用情况)

### **3.Optimization** 

Gradient Descent(在李宏毅老师课程中唯一用到的Optimization的方法).梯度下降的步伐取决于斜率绝对值的大小和η(learning rate).这一步等于η*偏导.
自己需要设定的东西叫hyperparameters和参数不一样.

停下来的条件:耐心没了(到达最大次数了)或者gradient = 0

Gradient Descent的问题：找不到全局的最小值(global minima),只能找到Local minima.Local minima问题只是一个幻觉,其实Gradient Descent面对的问题不是local minima.

微分一行代码就可以表达


---

上面只是在training data上面的loss,接下来要预测未来的数据,计算unseen during training的Loss.

画完图发现只是把曲线平移了一下而已.就是拿第一天的数据预测第二天的数据.数据上能说明Youtube的演算法

Domain Knowledge来修改Model.把wx1修改成了Σwjxj.训练资料上和没有看过的资料上Loss都降低了.


下面是重新定义一个新的function(Step 1)

---

Linear(线性) Models太过简单. 我们需要更复杂的模型.
可以改进成分段函数的形式.小于xx的时候取一个constant,大于xx的时候取一个constant.

All Piecewise Linear Curves = constant + sum of a set of 一些蓝色的function.

Beyond Piecewise Linear ?
取足够多的点连线变成Piecewise Linear Curve

可以用一条sigmoid function(S形函数)来逼近蓝色的function(Hard Sigmoid)(硬 S 形函数)

New Model:More Features  
**$y = b + \sum_{i} c_i*sigmoid(b_i + \sum_{j} w_{iJ}x_J)$**
本质是多个函数合在一块了( 加权 ＋ sigmoid ).其实还可以写成是矩阵相乘再相加.r = b + W叉乘x.   transpose(转置) y = b + cT叉乘a.   a = σ(b + W叉乘x )      feature:x   Unknown parameters:W  向量b  cT   b      把W的每一个color或者row拉成一个长的向量   再加上向量b cT b拼一起变成向量θ.

---

进入Step 2 和Step 3
L(θ)求Loss并且Optimization是一样的.
不一定把整个θ拿出来算Loss,还可以选一个batch,然后选gradient更新参数.再用下一个batch更新,以此类推...看完了所有的batch叫一个epoch，每次更新一次参数叫update.一个batch对应一个update.
Batchsize也是自己决定的(hyperparameter).

Hard Sigmoid可以看成两个Rectified Linear Unit(ReLU)(直接读就行)的相加.   c max(0, b + wx1)
可以把sigmoid换成ReLU.这两个都叫Activation function(激活函数)  ReLU比较好  ReLU的次数越多越准确,但是有可能有特出情况,比如过年啦!

Sigmoid/ReLU 叫Neuron,很多的Neuron叫Neural Network.新的名字: 每一排Neuron叫一个layer(hidden layer),很多hidden layer叫Deep,整套技术叫Deep Learning.

AlexNet:8 layers    VGG:19 layers   GoogleNet:22 layers   Residual Net:152 layers

为什么不做的更深?
**==training data 和 没看过的资料上结果不一致的情况== 叫 ==Overfitting==   选模型要选一个在没有看过的资料上表现好的**

Backpropagation:比较有效率的算gradient的方法,和上面讲的没有什么不同.
