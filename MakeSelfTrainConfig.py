import json
import os
import glob
from torch.cuda import is_available as cuda_available
from Code2D.ConfigTool import *
from Code2D.ConfigTool.ReadData import nifti_to_png
import random
from Default_config import get_defult_config

config = get_defult_config()
# get_defult_config()返回默认配置
# 包含model、loss、optimizer、scheduler、training、logging、early_stopping七个配置项

# 加入预训练模型参数地址
config['model']['pre_trained_model'] = r"/root/WangDao/2Dtraincode/record13/best_model.pth"

self_train_loss =   {
        "name": "CombinedLoss",
        "loss": [
                    {
                    "name": "ConfidenceThresholdLoss",
                    "weight": 1.0,
                    "args": {
                                "low_th": 0.1,
                                "high_th": 0.7,
                                "w_proj_neg": 1.0,
                                "w_pseudo": 3.0,
                                "sigmoid": True,
                            }
                    }
                ]
                    }

config['loss'] = self_train_loss

target_size = (400, 400)
imagetype = 'L'
labeltype = 'L'

# ------------------- 训练数据配置 -------------------
train_data_enhance = [
    {
        "type": "Resize",
        "params": {"size": target_size},
        "target": "both"
    },
    {
        "type": "RandomEqualize",
        "params": {
            "p": 1.0
        },
        "target": "image"
    },
    {
        "type": "RandomFlip",
        "params": {
            "p": 0.5,
            "direction": "horizontal"
        },
        "target": "both"
    },
    {
        "type": "RandomFlip",
        "params": {
            "p": 0.5,
            "direction": "vertical"
        },
        "target": "both"
    },
    {
        "type": "ToTensor",
        "target": "both"
    }
]

dataconfig = {
    "train": {
        "Use_dataenhance": True,        # 训练集禁用数据增强
        "data_per_picture": 5,          # 每张图片生成的训练样本数
        'preprocess': train_data_enhance,
        'dataset_type': "UnlabeledDataset",
        'picture_type': {
            'image': imagetype,
            'label': labeltype
        },
        'dataloader_args': {
            'batch_size': 8,
            'shuffle': True,
            'num_workers': 4
        }
    },
    "val": {
        "Use_dataenhance": False,       # 验证集禁用数据增强
        "data_per_picture": 1,          # 验证集保持原始数据（不增强）
        'preprocess': [
            {'type': 'Resize', 'params': {'size': target_size}, 'target': 'both'},
            {'type': 'RandomEqualize', 'params': {'p': 1.0}, 'target': 'image'},
            {'type': 'ToTensor', 'target': 'both'}
        ],
        'dataset_type': "UnlabeledDataset",
        'picture_type': {
            'image': imagetype,
            'label': labeltype
        },
        'dataloader_args': {
            'batch_size': 8,
            'shuffle': False,
            'num_workers': 2
        }
    },
    "test": {
        "Use_dataenhance": False,       # 测试集禁用数据增强
        "data_per_picture": 1,          # 测试集保持原始数据（不增强）
        "preprocess": [
            {'type': 'Resize', 'params': {'size': target_size}, 'target': 'image'},
            {'type': 'RandomEqualize', 'params': {'p': 1.0}, 'target': 'image'},
            {'type': 'ToTensor', 'target': 'image'}
        ],
        'picture_type': {
            'image': imagetype
        },
        'dataloader_args': {
            'batch_size': 1,
            'shuffle': False
        },
    }
}

# 动态生成数据路径
device = "cuda" if cuda_available() else "cpu"


# OCTA2D-selftrain
print("Start making Data...")
image_files = []
if device == "cuda":
    # data_path = r"C:\\Users\\Dell\\Desktop\\WDprogram\\OCTA2D"
    data_path = r"/root/WangDao/2Dtraincode/10025/Masked_volume.nii.gz"
    image_files = nifti_to_png(
        nifti_file_path=data_path,
        png_file_folder=getpath('Code2D','Temp'),
        cutting_dimension=1
    )
    label_path = r"/root/WangDao/2Dtraincode/10025/10025.bmp"
else:
    data_path = r"D:\WangDao\3DPreprocessCode\Processed_data\OCTA10010\Pictures"
    label_path = r"D:\WangDao\2DTrainCode\10010.bmp"

# image_dirs = [data_path]
# 读取并合并所有图像路径
# for d in image_dirs:
#     image_files.extend(sorted(glob.glob(os.path.join(d, "*.png"))))

# 打乱文件列表
combined_files = image_files
random.shuffle(combined_files)
image_files = combined_files

# 划分训练集和验证集（8:2）
# 训练集80%，验证集20%
# 标签统一使用二维投影标签
split_idx = int(len(image_files) * 0.9)
dataconfig['train']['data'] = [
    {'image': img, 'label': label_path} 
    for img in image_files[:split_idx]
]
dataconfig['val']['data'] = [
    {'image': img, 'label': label_path} 
    for img in image_files[split_idx:]
]
dataconfig['test']['data'] = []

  
config['data'] = dataconfig

# ------------------- 配置文件检查 --------------------

check_config(config)

# ------------------- 写入训练配置 --------------------
configpath = 'config.json'
with open(configpath, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)
print(f"config file saved to:\n{os.path.abspath(configpath)}")
