import matplotlib.pyplot as plt
# %matplotlib inline
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 每张图片是28*28*1的图片，我们需要将图片拉长成一个向量，这里暂时不关注图片的空间特征
num_inputs = 784
num_outputs = 10
# 正态分布初始化权重，均值为0，方差为0.01
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# 初始偏置置为0
b = torch.zeros(num_outputs, requires_grad=True)


def softmax(X):
    # 对每个元素进行指数计算
    X_exp = torch.exp(X)
    # 对每一行数据求和
    partition = X_exp.sum(1, keepdim=True)
    # 广播机制，对第i行都除以partition中的第i个元素，也就是除以每一行的和
    return X_exp / partition


def net(X):
    # X reshape成一个批次大小256*特征长度784的矩阵
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


# 交叉熵损失函数
def cross_entropy(y_hat, y):
    # range(len(y_hat))是生成一个从0到len大小的向量
    return -torch.log(y_hat[range(len(y_hat)), y])


# 将预测类别与真实y元素进行比较
def accuracy(y_hat, y):
    # 如果y_hat是二维矩阵，则按照每一行求argmax存到y_hat中
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# 评估在任意模型net的准确率，计算在指定数据集上模型的精度
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    # 这里的with nograd不能丢，丢了就训练不出来了
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# softmax回归训练
def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            # 这里需要取l.mean再前向传播
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]
    # for X, y in train_iter:
    #     y_hat = net(X)
    #     l = loss(y_hat, y)
    #     if isinstance(updater, torch.optim.Optimizer):
    #         updater.zero_grad()
    #         # 这里需要取l.mean再前向传播
    #         l.backward()
    #         updater.step()
    #         metric.add(float(l) * len(y), accuracy(y_hat, y), y.size().numel())
    #     else:
    #         l.sum().backward()
    #         updater(X.shape[0])
    #         metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # return metric[0] / metric[2], metric[1] / metric[2]


class Animator:  # @save
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        d2l.plt.draw()
        d2l.plt.pause(0.001)
        display.clear_output(wait=True)


# 开始训练调用
def train(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        # 在测试数据集上计算精度
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


lr = 0.1
num_epochs = 20


def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


def predict_ch3(net, test_iter, n=6):  # @save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
predict_ch3(net, test_iter)
d2l.plt.show()
