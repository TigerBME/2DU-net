import json
import os


def get_default_config(config_name: str) -> dict:
    """
    从同一目录下读取配置文件，删除部分配置，并返回处理后的配置。
    
    Args:
        config_name: 配置文件名称（如 'config_13.json'）
    
    Returns:
        删除了指定部分的配置字典
    """
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, config_name)
    
    
    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 删除指定的配置项
    # 1. 删除 logging 中的 log_dir
    if 'logging' in config and 'log_dir' in config['logging']:
        del config['logging']['log_dir']
    
    # 2. 删除 early_stopping 中的 model_save_path
    if 'early_stopping' in config and 'model_save_path' in config['early_stopping']:
        del config['early_stopping']['model_save_path']
    
    # 3. 删除 data.train.data 和 data.val.data
    if 'data' in config:
        if 'train' in config['data'] and 'data' in config['data']['train']:
            del config['data']['train']['data']
        if 'val' in config['data'] and 'data' in config['data']['val']:
            del config['data']['val']['data']
        if 'test' in config['data'] and 'data' in config['data']['test']:
            del config['data']['test']['data']
    
    # 4. 删除 statistics 整个部分
    if 'statistics' in config:
        del config['statistics']
    
    return config


