import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# 生成合成数据
n_data = torch.ones(100, 2)  # 创建一个 100x2 的全1的矩阵
class_0_data = torch.normal(2 * n_data, 1)  # 生成均值为 2*n_data，标准差为 1 的正态分布数据
class_0_labels = torch.zeros(100)  # 标签0，100个样本
class_1_data = torch.normal(-2 * n_data, 1)  # 生成均值为 -2*n_data，标准差为 1 的正态分布数据
class_1_labels = torch.ones(100)  # 标签1，100个样本

# 合并数据和标签
x = torch.cat((class_0_data, class_1_data), 0).type(torch.FloatTensor)  # 合并 class_0_data 和 class_1_data，类型转换为 FloatTensor
y = torch.cat((class_0_labels, class_1_labels), 0).type(torch.LongTensor)  # 合并 class_0_labels 和 class_1_labels，类型转换为 LongTensor

# 将数据和标签包装为 Variable
x, y = Variable(x), Variable(y)

# 可视化数据
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c = y.data.numpy(), s = 100, lw = 0, cmap = 'RdYlGn')
plt.show()


# 定义一个简单的神经网络模型
class SimpleNN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(SimpleNN, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层，输入特征数到隐藏层神经元数
        self.out = torch.nn.Linear(n_hidden, n_output)  # 输出层，隐藏层神经元数到输出类别数

    def forward(self, x):
        x = F.relu(self.hidden(x))  # 对隐藏层输出应用 ReLU 激活函数
        x = self.out(x)  # 通过输出层
        return x


# 实例化网络
net = SimpleNN(n_feature = 2, n_hidden = 10, n_output = 2)  # 创建一个具有 2 个输入特征，10 个隐藏神经元和 2 个输出的网络
print(net)  # 打印网络结构

# 定义优化器和损失函数
optimizer = torch.optim.SGD(net.parameters(), lr = 0.02)  # 使用随机梯度下降优化器
loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数，用于多分类问题

plt.ion()  # 开启交互模式，方便动态绘图

# 训练网络
for t in range(100):  # 训练 100 次
    out = net(x)  # 前向传播：通过网络计算输出
    loss = loss_func(out, y)  # 计算损失

    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播：计算梯度
    optimizer.step()  # 更新参数

    if t % 5 == 0:  # 每 5 次训练迭代更新一次图
        plt.cla()  # 清除当前图形
        prediction = torch.max(out, 1)[1]  # 预测输出的类别
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1],
                    c = prediction.data.numpy(),
                    s = 100, lw = 0, cmap = 'RdYlGn')  # 绘制散点图
        correct = torch.sum(torch.eq(prediction.data, y.data)).item()  # 计算正确预测的数量
        accuracy = correct / y.data.size(0)  # 计算准确度
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy,
                 fontdict = {'size': 20, 'color': 'red'})  # 显示准确度
        plt.pause(0.1)  # 暂停，以便动态显示图形

plt.ioff()  # 关闭交互模式
plt.show()  # 显示最终图形
