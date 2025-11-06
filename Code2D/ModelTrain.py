import json
import argparse
import datetime
from torch.cuda import is_available as cuda_available
from torch import device as torch_device

from DataLoader import get_dataloader
from Loss import get_loss_function
from Model import create_model
from Scheduler import make_schedule
from Train import train_model
from Stopping import get_manager
from Log import make_logger
from Optimizer import make_optimizer

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 将当前目录（project_root）加入Python路径

def load_config(config_path):
    """加载配置文件"""
    with open(config_path) as f:
        return json.load(f)

def main(config_path):
    # 加载配置
    config = load_config(config_path)
    
    # 设置设备
    device = torch_device('cuda' if cuda_available() else 'cpu')

    # 初始化日志系统
    logger_config = config.pop('logging')
    logger = make_logger(logger_config)
    print(f"Logging to {logger.log_dir}")
    
    # 加载数据
    data_config = config.pop('data')
    train_config = data_config.pop('train')
    val_config = data_config.pop('val')
    train_loader = get_dataloader(train_config, mode='train')
    val_loader = get_dataloader(val_config, mode='val')
    
    print(f"Training data: {len(train_loader.dataset)} samples")
    print(f"Validation data: {len(val_loader.dataset)} samples")
    
    # 创建模型
    model_config = config.pop('model')
    model = create_model(model_config).to(device)
    print(f"Model: {model.__class__.__name__}")
    
    # 初始化损失函数
    loss_config = config.pop('loss')
    criterion = get_loss_function(loss_config).to(device)
    print(f"Criterion: {criterion.__class__.__name__}")
    if criterion.__class__.__name__ == "CombinedLoss":
        print(' '*4, end='')
        for loss in loss_config['loss']:
            print(f"{loss['name']}: {loss['weight']} ",end='')
        print()
    
    # 创建优化器
    optimizer_config = config.pop('optimizer')
    optimizer = make_optimizer(model.parameters(), optimizer_config)
    print(f"Optimizer: {optimizer.__class__.__name__}")
    
    # 创建学习率调度器
    scheduler_config = config.pop('scheduler')
    scheduler = make_schedule(optimizer, scheduler_config)
    print(f"Scheduler: {scheduler.__class__.__name__}")
    
    # 初始化早停机制
    early_stopper_config = config.pop('early_stopping')
    early_stopper = get_manager(early_stopper_config)
    print(f"Early stopping: {early_stopper.__class__.__name__}")
    
    # 记录实验开始
    logger.start_experiment(cfg_path=config_path)
    
    # 执行训练
    print("Training...")
    train_config = config.pop('training')
    train_model(
        train_config=train_config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        earlystopper=early_stopper,
        scheduler=scheduler,
        logger=logger,
        device=device
    )
    
    # 记录实验结束
    logger.end_experiment(
        f"Best {early_stopper.mode} value: {early_stopper.best_score:.4f} "
        f"at epoch {early_stopper.best_epoch}"
    )

    # 验证模型

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--record', type=str, default=None,
                        help='Path to record file')
    args = parser.parse_args()
    print(f"Loading config from {args.config}")
    
    # 加载config检查创建时间
    with open(args.config) as f:
        config = json.load(f)
    create_time_str = config['statistics']['Creat_time']
    create_time = datetime.datetime.strptime(create_time_str, "%Y-%m-%d %H:%M:%S")
    current_time = datetime.datetime.now()
    time_diff = current_time - create_time
    
    if time_diff.total_seconds() > 300:  # 超过5分钟
        print(f"警告：config文件创建于{time_diff.seconds//60}分钟{time_diff.seconds%60}秒前")
        print("按Enter键继续运行，其他键退出...")
        user_input = input()
        if user_input != "":
            print("程序终止")
            exit()
    else:
        print(f"config为{time_diff.seconds//60}分钟{time_diff.seconds%60}秒之前创建，可以运行")
    
    record_path = args.record
    if record_path is not None:
        if not os.path.exists(record_path):
            print(f"Record_path {record_path} not exists, please build a record first.")
        else:
            print(f"Changing Record_path from {args.config['logging']['log_dir']} to {record_path}")
            args.config['logging']['log_dir'] = record_path

    main(args.config)