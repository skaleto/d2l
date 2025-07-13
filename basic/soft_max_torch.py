import matplotlib.pyplot as plt
# %matplotlib inline
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# PyTorch不会隐式的调整输入的形状
# 因此我们在线性层前定义了展平层flatten来调整网络输入形状
# nn.Linear(784, 10) 对应 softmax 分类器的核心原因在于PyTorch 的设计哲学：将线性变换与 softmax 激活分离，但在损失函数中隐式结合
# 为什么这样设计更合理？
# 灵活性：
# 在推理阶段，只需线性得分即可进行分类（选最大得分对应的类别），无需计算 softmax 概率。
# 如果需要概率输出（如计算置信度），可显式调用 nn.Softmax。
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


# 初始化权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


# 应用初始化权重
net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10

d2l.plt.show()
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

