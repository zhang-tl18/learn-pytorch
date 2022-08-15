import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 输入图像是单通道，conv1 kenrnel size=5*5，输出通道 6
        self.conv1 = nn.Conv2d(1, 6, 5)
        # conv2 kernel size=5*5, 输出通道 16
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # max-pooling 采用一个(2,2)的滑动窗口
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # 核(kernel)大小是方形的话，可仅定义一个数字，如 (2,2) 用 2 即可
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        # 除了 batch 维度外的所有维度
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == '__main__':
    net = Net()
    input = torch.randn(1, 1, 32, 32)

    # 定义伪标签
    target = torch.randn(10)
    # 调整大小，使得和 output 一样的 size
    target = target.view(1, -1)
    criterion = nn.MSELoss()
    
    # 创建优化器
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # 在训练过程中执行下列操作
    optimizer.zero_grad() # 清空梯度缓存
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    # 更新权重
    optimizer.step()
    