from datetime import datetime
import os
from Code2D.ConfigTool import getpath
from Code2D.Model import DEFAULT_CONFIG_ATTENTION_UNET as DEFAULT_MODEL_CONFIG

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
                "name": "ConfidenceThresholdLoss",
                "weight": 1.0,
                "args": {
                    "low_th": 0.3,
                    "high_th": 0.7,
                    "sigmoid": True,
                    "reduction": "mean"
                }
            }
        ]
    }
    config['loss'] = lossconfig

    # ------------------- 优化器配置 -------------------
    opconfig = {
        "name": "AdamW",
        "lr": 0.0001,
        "betas": [0.9, 0.999],
        "eps": 1e-08,
        "weight_decay": 1e-4
    }
    config['optimizer'] = opconfig


    # ------------------- 学习率调度器配置 -------------------
    schedulerconfig = {
        "name": "PolyLR",            # ← 改为 poly
        "power": 0.9,                # ← poly 标准参数
        "max_epoch": 10,            # ← poly 需要总 epoch 数
    }

    config['scheduler'] = schedulerconfig

    # ------------------- 训练参数配置 -------------------
    trainconfig = {
        "epochs": 10,                # ← 设置为约 10 个 epoch
        "accumulation_steps": 1,
        "warm_up_epochs": 0,         # ← poly 通常不使用 warmup，故改为0
        "train_kind": "selftrain",
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
            'sensitivity': False, 
            'specificity': False, 
            'accuracy': False,
            'mean_entropy': True,
            'mean_confidence': True,
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

