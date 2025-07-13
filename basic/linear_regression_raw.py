import matplotlib.pyplot as plt
# %matplotlib inline
import random
import torch
from d2l import torch as d2l


# 人工模拟生成数据
def synthetic_data(w, b, num_examples):
    # 生成均值为0，方差为1的随机数，样本数=num_examples，列数=w的个数
    X = torch.normal(0, 1, (num_examples, len(w)))
    # y=X*w+b
    y = torch.matmul(X, w) + b
    # y加上一个均值为0，方差为0.01的噪音
    y += torch.normal(0, 0.01, y.shape)
    # reshape成列向量返回
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
# features的每一行都是一个二维的数据样本，labels的每一行都是一个一维的标签值
features, labels = synthetic_data(true_w, true_b, 1000)


# 返回一批随机的样本
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 打乱索引，用于随机访问
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        # yield关键字使函数成为迭代器，返回数据但保留函数状态
        yield features[batch_indices], labels[batch_indices]


# 定义初始化模型参数
# 输入维度是2，所以w是一个长度为2的向量，初始化成一个均值为0，方差为0.01的
# 因为要计算梯度，所以requires_grad为True
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 定义线性回归模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b


# 定义损失函数-均方误差
def squared_loss(y_hat, y):
    # 因为形状可能不一样，所以reshape成一样
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法-小批量随机梯度下降
def sgd(params, lr, batch_size):
    # 更新的时候无需计算梯度，因为训练时在调用sgd前一句使用backward计算了梯度
    with torch.no_grad():
        for param in params:
            # 均方误差里没有除均值，放哪边都可以。放这边可以用batch_size来归一化步长，使得结果去除batch_size的影响
            param -= lr * param.grad / batch_size
            # 更新完成后梯度设为0，防止对下一次产生影响
            param.grad.zero_()


# 训练过程
# 指定超参数
batch_size = 10
# 学习率太大可能会出现梯度爆炸，太小会迭代太慢
lr = 0.03
# 学习次数
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # 求x和y的小批量损失
        l = loss(net(X, w, b), y)
        # 因为l形状是(batch_size, 1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        # 使用参数的梯度更新参数
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

# d2l.set_figsize()
# d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# plt.show()
