class DefaultConfig(object):

    op_train = 'train'                      # 训练操作
    op_val = 'val'                          # 验证操作
    op_test = 'test'                       # 测试操作
    
    visenv = 'default'                      # Visdom环境
    
    train_data_root = './dataset/train'     # 训练集的存放路径
    test_data_root = './dataset/test'       # 测试集的存放路径
    load_model_root = './checkpoints'                  # 加载预训练模型的路径，None表示不加载
    save_model_root = './checkpoints'       # 模型的保存路径

    batch_size = 32                         # batch_size的大小
    device = 'cuda'                         # 是否使用GPU加速
    num_workers = 0                         # 加载数据时的线程数
    print_freq = 20                         # 打印信息的间隔伦次

    epoch = 30                              # 训练轮数
    lr = 0.001                              # 初始化学习率
    lr_decay = 0.95                         # 学习衰减率，lr = lr * lr_decay
    weight_decay = 1e-4                     # 权重衰减率

