# train/trainer.py
import torch
import time
from tqdm import tqdm
from Stopping import EarlyStopper
from torch.utils.data import DataLoader

def train_model(train_config: dict, model: torch.nn.Module, 
                train_loader: DataLoader, val_loader: DataLoader, 
                criterion: torch.nn.modules.loss, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler,
                earlystopper: EarlyStopper, logger,
                device: torch.device):
    """
    核心训练模块
    Args:
        config: 包含所有训练参数的配置字典
        model: 待训练的模型
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        earlystopper: 早停器
        logger: 日志记录器
        device: 训练设备
        accumulation_steps: 每多少次前向传播之后进行一次反向传播
    """
    # 从配置中获取参数
    must_keywords = ['epochs','accumulation_steps']
    for k in must_keywords:
        assert k in train_config, f"Missing key {k} in training config"
    epochs = train_config['epochs']
    accumulation_steps = train_config['accumulation_steps']
    warm_up_epochs = train_config.get('warm_up_epochs', 0)
    base_lr = optimizer.param_groups[0]['lr']    
    
    # 训练循环
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        
        # 学习率热身阶段
        if warm_up_epochs and epoch < warm_up_epochs:
            _warmup_lr(optimizer=optimizer, base_lr=base_lr, 
                       epoch=epoch, total=warm_up_epochs)
        
        # 训练阶段
        train_loss, lr_value = _train_epoch(
            model, train_loader, criterion, optimizer, device, accumulation_steps
        )
        
        # 验证阶段
        val_metrics = _validate_model(
            model, val_loader, criterion, device
        )
        
        # 记录指标
        epoch_time = time.time() - epoch_start
        log_data = {
            'epoch': epoch+1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'lr': lr_value,
            'time': epoch_time,
            **val_metrics['metrics']
        }
        logger.log_metrics(log_data)
        print(f"Dice = {val_metrics['metrics']['dice']}")

        # 检查早停和学习率调整
        early_stop_return_config =  earlystopper.check_early_stop(val_metrics['metrics']['dice'], model)
        # print(f"Score is {early_stop_return_config.pop('best_score')}")
        print(f"the train will early stop in {early_stop_return_config.pop('patience_left')} epochs")
        print(f"the progress is {early_stop_return_config.pop('progress')}")

        if early_stop_return_config.pop('early_stop', False):
            print(f"early stop triggered, training stopped at epoch {epoch+1}")
            break    

        # 更新学习率（根据不同调度器类型区分）
        if scheduler and epoch >= warm_up_epochs:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau 需要传入验证集指标
                scheduler.step(val_metrics['metrics']['dice'])  
                # scheduler.step(val_metrics['loss'])  
            else:
                # 其他调度器只需简单调用 step()
                scheduler.step()


def _train_epoch(model, 
                 loader, 
                 criterion, 
                 optimizer, 
                 device, 
                 accumulation_steps: int):
    """执行单个epoch的训练"""
    frontlength = 25
    backlength = 15

    total_loss = 0
    current_lr = optimizer.param_groups[0]['lr']
    accumulated_grad = 0  # 用于记录累积的梯度

    description = f"Training (LR={current_lr:.2e})"
    progress_bar = tqdm(loader, desc=description + (frontlength-len(description))*" ")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss = loss / accumulation_steps  # 根据累积步数调整损失
        loss.backward()
        
        total_loss += loss.item()

        accumulated_grad += 1  # 记录累积的步数

        if accumulated_grad % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            accumulated_grad = 0  # 重置累积的步数

        postfix_str = f"loss={loss.item():.4f}"
        progress_bar.set_postfix_str(postfix_str + (backlength-len(postfix_str))*" ")
    
    return total_loss/len(loader), current_lr

def _validate_model(model, loader, criterion, device):
    """执行模型验证"""

    frontlength = 25
    backlength = 17

    model.eval()
    total_loss = 0
    metrics = {'dice': 0, 'sensitivity': 0, 'specificity': 0, 'accuracy': 0}
    
    with torch.no_grad():
        description = "Validating"
        progress_bar = tqdm(loader, desc=description + (frontlength-len(description))*" ")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # 计算指标
            batch_metrics = _calculate_metrics(outputs, targets)
            for k in metrics:
                metrics[k] += batch_metrics[k] * inputs.size(0)

            postfix_str = f""
            progress_bar.set_postfix_str(postfix_str + (backlength-len(postfix_str))*" ")
    
    # 计算平均指标
    num_samples = len(loader.dataset)
    return {
        'loss': total_loss/len(loader),
        'metrics': {k: metrics[k]/num_samples for k in metrics}
    }

def _calculate_metrics(pred, target, threshold=0.5):
    """计算评估指标"""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = (target > threshold).float()
    
    tp = (pred * target).sum()
    tn = ((1-pred) * (1-target)).sum()
    fp = (pred * (1-target)).sum()
    fn = ((1-pred) * target).sum()
    
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    dice = 2*tp / (2*tp + fp + fn + 1e-8)
    
    return {
        'dice': dice.item(),
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item(),
        'accuracy': accuracy.item()
    }

def _warmup_lr(optimizer, base_lr, epoch, total):
    """线性学习率热身"""
    lr = base_lr * (epoch + 1) / total
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
