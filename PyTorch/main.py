import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

# 生成合成数据
inputs = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1)  # 创建一个包含100个点的列向量，范围从-1到1
targets = inputs.pow(2) + 0.2 * torch.rand(inputs.size())  # 使用二次关系生成目标值，并添加一些无关的

# 绘制合成数据
plt.scatter(inputs.data.numpy(), targets.data.numpy())  # 绘制inputs与targets的散点图
plt.show()  # 显示图像


# 定义一个简单的神经网络模型
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)  # 隐藏层
        self.output_layer = nn.Linear(hidden_size, output_size)  # 输出层

    def forward(self, x):
        x = F.relu(self.hidden_layer(x))  # 应用ReLU激活函数
        x = self.output_layer(x)  # 通过输出层
        return x


# 初始化神经网络
model = SimpleNet(1, 10, 1)  # 1个输入特征，10个隐藏单元，1个输出特征
print(model)  # 打印网络架构

plt.ion()  # 开启交互模式以便实时更新
plt.show()  # 显示图像窗口

# 设置优化器和损失函数
optimizer = torch.optim.SGD(model.parameters(), lr = 0.5)  # 使用随机梯度下降优化器
loss_function = nn.MSELoss()  # 使用均方误差损失函数

# 训练循环
for epoch in range(100):
    predictions = model(inputs)  # 前向传播

    loss = loss_function(predictions, targets)  # 计算损失

    optimizer.zero_grad()  # 清除梯度
    loss.backward()  # 反向传播以计算梯度
    optimizer.step()  # 更新权重

    if epoch % 5 == 0:  # 每5步，更新一次图像
        plt.cla()  # 清除当前图像
        plt.scatter(inputs.data.numpy(), targets.data.numpy())  # 绘制原始数据
        plt.plot(inputs.data.numpy(), predictions.data.numpy(), 'r-', lw = 5)  # 绘制模型的预测结果
        plt.text(0.5, 0, 'Loss = %.4f' % loss.item(), fontdict = {'size': 20, 'color': 'red'})  # 显示当前损失
        plt.pause(0.5)  # 暂停以更新图像

plt.ioff()  # 关闭交互模式
plt.show()  # 显示最终图像
