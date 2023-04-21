import torch as t
from torch import nn
import time as time


# 封装了 nn.Module，主要提供 load 和 save 两个方法
class BasicModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model_name = str(type(self))   # 模型的默认名称

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        if name is None:
            prefix = './checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H%M%S.pth')
        t.save(self.state_dict(), name)
