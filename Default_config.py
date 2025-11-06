from datetime import datetime
import os
from Code2D.ConfigTool import getpath
from Code2D.Model import DEFAULT_CONFIG_UNET as DEFAULT_MODEL_CONFIG

def get_defult_config():
    imagetype = 'L'
    labeltype = 'L'

    config = dict()
    # ------------------- 模型配置 -------------------
    modelconfig = DEFAULT_MODEL_CONFIG
    modelconfig['in_channels'] = (1 if imagetype == 'L' else 3)

    assert modelconfig['spatial_dims'] == 2, "only support 2D model"
    assert modelconfig['out_channels'] == 1, "only support binary segmentation"

    config['model'] = modelconfig

    # ------------------- 损失函数配置 -------------------
    lossconfig = {
        "name": "CombinedLoss",
        "loss": [
            {
                "name": "DiceLoss",
                "weight": 0.9,
                "args": {
                    "smooth": 1e-5,
                    "sigmoid": True,
                    "reduction": "mean"
                }
            },
            {
                "name": "BCELoss",
                "weight": 0.1,
                "args": {
                    "reduction": "mean",
                    "pos_weight": 5
                }
            }
        ]
    }
    config['loss'] = lossconfig

    # ------------------- 优化器配置 -------------------
    opconfig = {
        "name": "AdamW",
        "lr": 0.001,             # 保持不变，AdamW对lr不敏感
        "betas": [0.9, 0.999],   # 一般不改
        "eps": 1e-08,
        "weight_decay": 1e-4     # 核心变化：启用权重衰减
    }
    config['optimizer'] = opconfig


    # ------------------- 学习率调度器配置 -------------------
    schedulerconfig = {
        "name": "ReduceLROnPlateau",
        "mode": "max",
        "factor": 0.6,
        "patience": 7,
        "threshold": 0.001,
        "cooldown": 1,
        "threshold_mode": "abs",
        "min_lr": 1e-6,
        "eps": 1e-8
    }

    config['scheduler'] = schedulerconfig

    # ------------------- 训练参数配置 -------------------
    trainconfig = {
        "epochs": 50,
        "accumulation_steps": 1,
        "warm_up_epochs": 3,
    }
    config['training'] = trainconfig

    # ------------------- 日志配置 -------------------
    log_path = getpath('Records','record')
    logconfig = {
        "log_dir": os.path.join("./", log_path),
        "metrics_to_log": 
        {
            'epoch':True,
            'train_loss':True,
            'val_loss':True,
            'dice':True,
            'lr':True,
            'time':True,
            'sensitivity': True, 
            'specificity': True, 
            'accuracy': True
        },
        "save_best_only": True,
        "overwrite": False
    }
    config['logging'] = logconfig

    # ------------------- 早停机制配置 -------------------
    earlystopconfig = {
        "patience": 20,
        "delta": 0.0008,
        "mode": "max",
        "model_save_path": os.path.join(log_path, "best_model.pth")
    }
    config['early_stopping'] = earlystopconfig

    return config

if __name__ == '__main__':
    config = get_defult_config()