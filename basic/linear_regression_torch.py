import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 使用pytorch现有API进行数据读取
def load_array(data_arrays, batch_size, is_train=True):
    # *表示对参数解包，表示一个一个调用
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array([features, labels], batch_size)

# 模型定义，Linear其实就是全连接层，参数2和1代表输入和输出的特征数量
net = nn.Sequential(nn.Linear(2, 1))
# 随机初始化权重
# normal_表示使用正态分布来替换值
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

lr = 0.03
# 均方误差，或者L2范数
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# 训练过程
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        # 优化器梯度清零
        trainer.zero_grad()
        # 计算backward，pytorch内部已经做了sum
        l.backward()
        # step进行模型更新
        trainer.step()
    # 迭代一次后看一下loss结果
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1},loss {l:f}')
