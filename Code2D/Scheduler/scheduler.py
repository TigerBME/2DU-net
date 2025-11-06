import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Optimizer
import sys
from .PolyLR import PolyLR

current_module = sys.modules[__name__]

def make_schedule(optimizer: Optimizer, scheduler_config: dict):
    """
    根据配置文件动态创建学习率调度器，无需使用if-else条件判断

    Args:
        optimizer (torch.optim.Optimizer): 已初始化的优化器实例
        scheduler_config (dict): 包含调度器配置的字典，格式示例:
            {
                "scheduler": {
                    "name": "StepLR",
                    "step_size": 10,
                    "gamma": 0.1
                }
            }

    Returns:
        torch.optim.lr_scheduler._LRScheduler: 实例化的学习率调度器

    Raises:
        ValueError: 当配置中指定的调度器名称不存在时
        TypeError: 当参数与调度器构造函数不匹配时
    """
    scheduler_name = scheduler_config.pop("name")
    
    try:
        # 首先尝试在当前模块中获取调度器类
        scheduler_class = getattr(current_module, scheduler_name)
    except AttributeError:
        try:
            # 如果当前模块中找不到，则尝试在torch.optim.lr_scheduler中获取
            scheduler_class = getattr(lr_scheduler, scheduler_name)
        except AttributeError:
            raise ValueError(f"Unsupported scheduler: '{scheduler_name}'. "
                             f"Available schedulers in {current_module.__name__}: {dir(current_module)} | "
                             f"Available schedulers in torch.optim.lr_scheduler: {dir(lr_scheduler)}")

    # 提取构造函数参数（已排除'name'键）
    params = scheduler_config
    
    try:
        # 实例化调度器
        return scheduler_class(optimizer, **params)
    except TypeError as e:
        # 捕获参数错误并增强错误信息
        required_params = scheduler_class.__init__.__code__.co_varnames
        raise TypeError(
            f"Invalid parameters for {scheduler_name}. "
            f"Required parameters: {required_params[1:]} | "
            f"Given parameters: {list(params.keys())}"
        ) from e
