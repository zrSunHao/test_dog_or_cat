# pre1 导入所需的类、函数
from torch.utils.data import DataLoader
import torch as t
import os

from data import DogCat
from config import DefaultConfig
from models import Net2, CNet

# pre2 实例化配置类
cfg = DefaultConfig()

# step1 实例化模型
model_ft = Net2()
model_path = cfg.load_model_root + '/pre_60_0.857.pth'
assert os.path.exists(model_path)
state_dict = t.load(model_path)
model_ft.load_state_dict(state_dict)
model_ft.to(cfg.device)

# step2 预处理数据，加载数据
test_dataset = DogCat(root=cfg.test_data_root, mode=cfg.op_test)
testloader = DataLoader(test_dataset,
                        batch_size=cfg.batch_size,
                        shuffle=False,
                        num_workers=cfg.num_workers
                        )

classes = ['cat','dog']
# step3 定义测试函数
def test(model, testloader):
    model.eval()
    with t.no_grad():
        for idx, (data, names) in enumerate(testloader):
            input = data.to(cfg.device)
            names = names.to(cfg.device)
            output = model(input)
            predicted = t.max(output.data, 1).indices               # 每行最大值的索引
            print(names)
            for i,p in enumerate(predicted):
                print(i+1,classes[p])
            break

test(model_ft,testloader)