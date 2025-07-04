# Deep Reinforcement Learning(RL)
Introduction of Deep Reinforcement Learning
Supervised Learning -> RL
比如下围棋，如何去找一个最好的答案.收集答案困难 + 人们无法标注 : RL
## RL和ML一样都是三个步骤
Machine Learning = Looking for a Function
Actor Environment进行互动 Environment 对 Actor(function)进行Observation(input) Actor进行Action(output).
Environment 对 Actor Reward. Find a policy maximizing total reward
Example: Playing Video Game ( Space invader )   Learning to play Go 
Machine Learning
Step 1: function with unknown 
Step 2: define loss from training data
Step 3: optimization

## RL
## Step 1: function with unknown    function:Policy Network(Actor) 输入是a vector or a pixels 输出是每一个行为
实际上这就是Classification Task!!!  图片? 也许CNN.作业中是FC.  采取Sample based on scores.
## Step 2: Define "Loss" 从开始直到游戏结束, Total reward(return).一次的叫reward 正常的叫return.想让Total reward最大.把-Total reward当成Loss 越小越好.
## Step 3: Optimization    
Trajectory t = {s1,a1,s2,a2,...}  Reward (function)要看s1和a1得到r1  把所有的r加起来得到R(t) (Return)  Optimization要找到一个Network参数放在Actor里面可以让R(t)越大越好.
a1是sample产生的.这不是一个普通的Network,具有随机性.Env 和 Reward不是Network,只是一个黑盒子.Reward是一条规则.Reward(有些)和Env都具有随机性.

RL最主要的问题是 optimization. c.f.GAN   GAN又干了!   只不过这里面的Reward,Env不是Network.所以没有办法用Gradient Descent.


# Policy Gradient
让机器学习看到xx情况下xx本质上就是分类问题:把预期结果定义为label,然后计算输出和label的Cross-entropy,让他最小.如果让他不干xx,就取个负号即可. 
还可以计算L = e1 - e2. 很像train一个classifier,控制Actor的行为.


每一个action都会影响后面的rewards.Reward delay:有时候需要牺牲现在的reward来获得long-term reward

## Version 1
G1 = r1 + r2 + r3 + ... + rN 来评估a1的好坏    G2 = r2 + r3 + ... + rN 来评估a2的好坏 ......
G:cumulated reward
Version 1的问题在于可能会抢功劳！

## Version 2 
Discout facctor γ < 1   G1' = r1 + γr2 + γ2r3 + ...  距离越远γ平方越多
A1 = G1' A2 = G2' ......   
到这里已经合理多了

不同的RL方法是在A上面下文章

## Version 3
Good or bad reward is "relative"
If all the rn >= 10   rn = 10 is negative...   reward是相对的  
我们需要做标准化 所有的G' 减去b    B:baseline
如何设定 baseline b ?

## Policy Gradient
Initialize actor network parameters θ0
For training iteration i = 1 to T
 Using actor θi-1 to interact
  Obtain data {s1,a1},{s2,a2},...,{sN,aN}
  Compute A1,A2,..,AN  (这里面一定要改)
  Compute loss L
  θi = θi-1 - η * grad(L)  和gradient descent是一样的
一般的training data collection 都是在training 之外的，但是RL的data collection是在training循环里面的,所以非常费时间
同一个Action对于不同的Actor的作用效果是不一样的,所以上面的资料只能训练θi-1，不能训练θi.因此就需要不断更新data. e.g.棋魂中大马步飞和小马步飞 

上面的是On-policy:train和interact的Actor是同一个    Off-policy:要训练的Actor 和 与环境互动的Actor是两个.这样就不用在每个epoch收集资料了. 经典的做法Proximal Policy Optimization(PPO).重点是train的Actor要知道自己和interact的Actor的difference.interact的actor的行为有些可以采纳,有些不行

Exploration(训练过程中非常重要的技巧)：data collection里面具有随机性(随机性十分重要). e.g.Enlarge output entropy  Add noises onto parameters.

DeepMind - PPO (和OpenAI同时提出)
应用于一些机器人.


Critic: Given actor θ,how good it is when observing s (and taking action a)
Value function Vθ(s): When using actor θ,the discounted cumulated reward expects to be obtained after seeing s
Value function的数值和观察的actor有关系.
Monte-Carlo(MC) based approach:玩了很多场游戏.看到s就知道Vθ了
Temporal-differnce(TD) approach:不用玩完正常游戏,只需要上下一点就可以更新Vθ的参数.

##  Version 4
Advantage Actor-Critic

# RL很吃运气,很看sample的怎么样
