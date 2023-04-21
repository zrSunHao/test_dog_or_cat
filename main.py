# pre1 导入所需的类、函数
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch as t
import os

from data import DogCat
from config import DefaultConfig
from models import Net2, CNet

# pre2 实例化配置类
cfg = DefaultConfig()

# step1 实例化模型
model_ft = Net2()
model_path = cfg.load_model_root + '/pre_7_0.857.pth'
if os.path.exists(model_path):
    state_dict = t.load(model_path)
    model_ft.load_state_dict(state_dict)
model_ft.to(cfg.device)

# step2 预处理数据，加载数据
train_dataset = DogCat(root=cfg.train_data_root, mode=cfg.op_train)
trainloader = DataLoader(train_dataset,
                         batch_size=cfg.batch_size,
                         shuffle=True,
                         num_workers=cfg.num_workers
                         )
val_dataset = DogCat(root=cfg.train_data_root, mode=cfg.op_val)
valloader = DataLoader(val_dataset,
                       batch_size=cfg.batch_size,
                       shuffle=True,
                       num_workers=cfg.num_workers
                       )


# step3 定义优化器、损失函数
optimizer = optim.SGD(model_ft.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
criterion = nn.CrossEntropyLoss().to(cfg.device)


# step4 定义训练函数
def train_print(epoch, num, idx, loss_item, acc_sum):
    loss_print = format(loss_item, '.5f')
    acc_ratio = acc_sum / (num*cfg.batch_size)
    acc_print = format(acc_ratio*100, '.2f')
    epo = str(epoch).ljust(2, " ")
    batch_print = format(((idx+1))/len(trainloader)*100, '.0f')
    msg = f'epoch:{epo}/{cfg.epoch}  batch:{batch_print}%  loss:{loss_print}  accuracy:{acc_print}%'
    print(msg)

def train(model, trainloader, epoch):
    model.train()
    acc_sum = 0
    loss_sum = 0
    print_freq = int(len(trainloader)/10)
    num = 0
    for idx, (data, labels) in enumerate(trainloader):
        input = data.to(cfg.device)
        target = labels.to(cfg.device)
        optimizer.zero_grad()                                   # 梯度清零

        output = model(input)
        loss = criterion(output, target)                        # 计算误差
        loss.backward()                                         # 反向传播
        optimizer.step()                                        # 更新参数

        print_loss = loss.data.item()                           # 损失值
        loss_sum += print_loss

        predicted = t.max(output.data, 1).indices               # 每行最大值的索引
        acc_pred = t.sum(predicted == target).data               # 预测正确的个数
        acc_sum += acc_pred.tolist()

        num += 1
        if (idx+1) % print_freq == 0 or idx == len(trainloader) - 1:
            loss_item = loss.data.item()
            train_print(epoch, num, idx, loss_item, acc_sum)
            num = 0
            loss_sum = 0
            acc_sum = 0


# step5 定义验证函数
def val_print(epoch, num, loss_sum, acc_sum):
    loss_ave = loss_sum/num
    loss_print = format(loss_ave, '.5f')
    acc_ratio = acc_sum / (num*cfg.batch_size)
    acc_print = format(acc_ratio*100, '.2f')
    epo = str(epoch).ljust(2, " ")
    msg = f'epoch:{epo}/{cfg.epoch}  ==========>  loss:{loss_print}  accuracy:{acc_print}%'
    print(msg)
    return acc_ratio

def val(model, valloader, epoch):
    model.eval()
    acc_sum = 0
    loss_sum = 0
    num = len(valloader)
    with t.no_grad():
        for idx, (data, labels) in enumerate(valloader):
            input = data.to(cfg.device)
            target = labels.to(cfg.device)
            output = model(input)

            loss = criterion(output, target)                        # 计算误差
            loss_sum += loss

            predicted = t.max(output.data, 1).indices               # 每行最大值的索引
            acc_pred = t.sum(predicted == target).data              # 预测正确的个数
            acc_sum += acc_pred.tolist()
    acc_ratio = val_print(epoch, num, loss_sum, acc_sum)
    return acc_ratio


# 测试模型
def test(model, testloader, epoch):
    for ii, (data, label) in enumerate(testloader):
        if (ii > 10):
            break
        print(label)


# step6 循环，执行指定次数的epoch，保存精确度高的模型
max_acc = 0
for epo in range(cfg.epoch):
    train(model_ft, trainloader, epo + 1)
    acc = val(model_ft, valloader, epo + 1)
    print('\n')
    if (max_acc < acc):
        # max_acc = acc
        name = 'model_' + str(epo+1) + '_' + str(round(acc, 3)) + '.pth'
        t.save(model_ft.state_dict(), f'{cfg.save_model_root}/{name}')  # 保存模型
