# parlgymgame

程序是在https://github.com/PaddlePaddle/PARL/tree/develop/examples/DQN_variant基础上进行了修改，可以不需使用atari的rom，直接gym atari游戏的名字既可以使用，gym游戏的名字在gym官网可以找到。

训练使用实例 python train.py "Phoenix-v0" 

第一个参数为gym game 的名字，可以在gym官网查找游戏的名字直接替换即可，第二个参数为算法选择，DQN/DDQN/Dueling，第三个参数训练次数。

测试使用实例 python test.py "Phoenix-v0" 

