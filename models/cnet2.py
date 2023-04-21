from torch import nn
from torch.nn import functional as F

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)   # 3个颜色通道 输出维度 卷积核大小
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)      # 类别数为10

    def forward(self, x):
        # 特征提取，第 1 层
        x = self.conv1(x)               # 卷积
        x = F.relu(x)                   # 激活
        x = F.max_pool2d(x, (2, 2))     # 池化
        # 特征提取，第 2 层
        x = self.conv2(x)               # 卷积
        x = F.relu(x)                   # 激活
        x = F.max_pool2d(x, 2)          # 池化
        
        x = x.view(x.size(0), -1)     # 改变 Tensor 的形状，-1表示自适应
        x = self.fc1(x)                 # 全连接，第 1 层
        x = F.relu(x)
        x = self.fc2(x)                 # 全连接，第 2 层
        x = F.relu(x)
        x = self.fc3(x)                 # 全连接，第 3 层
        return x                        # 返回网络的输出结果
