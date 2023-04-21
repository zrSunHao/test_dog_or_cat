from torch import nn
from torch.nn import functional as F


class CNet(nn.Module):

    def __init__(self):
        super().__init__()
        # 第一层卷积 输入层  输入3通道图片，输出32通道，卷积核大小为3，步长为1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, stride=1, padding=0),
            nn.ReLU(),                      # 激活函数
            nn.MaxPool2d(kernel_size=2),    # 最大池化
        )
        # 第二层卷积 中间层  输入32通道图片，输出64通道，卷积核大小为3，步长为1
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=0),
            nn.ReLU(),                      # 激活函数
            nn.MaxPool2d(kernel_size=2),    # 最大池化
        )
        # 第三层卷积  中间层  输入64通道图片，输出128通道，卷积核大小为3，步长为1
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=0),
            nn.ReLU(),                      # 激活函数
            nn.MaxPool2d(kernel_size=2),    # 最大池化
        )
        # 第四层卷积  中间层  输入128通道图片，输出128通道，卷积核大小为3，步长为1
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=1, padding=0),
            nn.ReLU(),                      # 激活函数
            nn.MaxPool2d(kernel_size=2),    # 最大池化
        )
        # 输出分类信息
        self.output = nn.Linear(in_features=128 * 12 * 12, out_features=2)

    # 前向传播，依次计算
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        temp = out.view(x.size(0),12 * 12 * 128)
        res = self.output(temp)
        return res
