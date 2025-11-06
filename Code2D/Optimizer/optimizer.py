import torch.optim as optim

def make_optimizer(model_params: iter, optimizer_config: dict) -> optim.Optimizer:
    """
    根据配置文件动态创建优化器，无需使用if-else条件判断

    Args:
        model_params (iterable): 模型参数（通常来自于model.parameters()）
        optimizer_config (dict): 包含优化器配置的字典，格式示例:
            {
                "optimizer": {
                    "name": "Adam",
                    "lr": 0.001,
                    "betas": (0.9, 0.999)
                }
            }

    Returns:
        torch.optim.Optimizer: 实例化的优化器

    Raises:
        ValueError: 当配置中指定的优化器名称不存在时
        TypeError: 当参数与优化器构造函数不匹配时
    """
    optimizer_name = optimizer_config.pop("name")
    
    try:
        # 动态获取优化器类
        optimizer_class = getattr(optim, optimizer_name)
    except AttributeError:
        raise ValueError(f"Unsupported optimizer: '{optimizer_name}'. "
                         f"Available optimizers: {dir(optim)}")

    # 提取构造函数参数（已排除'name'键）
    params = optimizer_config
    
    try:
        # 实例化优化器
        return optimizer_class(model_params, **params)
    except TypeError as e:
        # 捕获参数错误并增强错误信息
        required_params = optimizer_class.__init__.__code__.co_varnames
        raise TypeError(
            f"Invalid parameters for {optimizer_name}. "
            f"Required parameters: {required_params[1:]} | "
            f"Given parameters: {list(params.keys())}"
        ) from e
