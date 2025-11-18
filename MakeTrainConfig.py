import json
import os
import glob
from torch.cuda import is_available as cuda_available
from Code2D.ConfigTool import *
import random
from Default_config import get_defult_config

config = get_defult_config()
# get_defult_config()返回默认配置
# 包含model、loss、optimizer、scheduler、training、logging、early_stopping七个配置项


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
        "type": "RandomRotate",
        "params": {
            "p": 0.6,
            "max_angle": 30
        },
        "target": "both"
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
        "type": "RandomMask",
        "params": {
            "p_large": 0.3,
            "p_small": 0.2,
            "angle_range": [0, 360],
            "move_range": [170, 180],
            "radius_range": [320, 340]
        },
        "target": "both"
    },
    {
        "type": "ToTensor",
        "target": "both"
    },
    {
        "type": "AddGaussianNoise",
        "params": {
            "p": 0.5,
            "mean": 0,
            "var": 0.01
        },
        "target": "image"
    }
]

dataconfig = {
    "train": {
        "Use_dataenhance": True,        # 训练集启用数据增强
        "data_per_picture": 10,          # 每张图片生成的训练样本数
        'preprocess': train_data_enhance,
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
            {"type": "RandomMask","params": 
                {
                    "p_large": 0.3,
                    "p_small": 0.2,
                    "angle_range": [0, 360],
                    "move_range": [170, 180],
                    "radius_range": [320, 340]
                },
                "target": "both"
            },
            {'type': 'ToTensor', 'target': 'both'}
        ],
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


# OCTA2D
print("Start making Data...")
if device == "cuda":
    # data_path = r"C:\\Users\\Dell\\Desktop\\WDprogram\\OCTA2D"
    data_path = r"/root/WangDao/Data"
else:
    data_path = r"D:\\C_File\\WDdata\\OCTA2D"


# 无卡开机时，直接指定数据路径
# data_path = r"D:\\C_File\\WDdata\\OCTA2D"
# data_path = r"/root/WangDao/Data"


# 定义四个子目录
image_dirs = [os.path.join(data_path, "images"),
              #os.path.join(data_path, "Moreimages")
              ]

label_dirs = [os.path.join(data_path, "labels"),
              #os.path.join(data_path, "Morelabels")
              ]

# 读取并合并所有图像路径
image_files = []
for d in image_dirs:
    image_files.extend(sorted(glob.glob(os.path.join(d, "*.bmp"))))

# 读取并合并所有标签路径
label_files = []
for d in label_dirs:
    label_files.extend(sorted(glob.glob(os.path.join(d, "*.bmp"))))

'''
# DRIVE
data_path = r"C:\\Users\\Dell\\Desktop\\WDprogram\\DRIVE"

image_files1 = sorted(glob.glob(os.path.join(data_path, "training\\images", "*.tif")))
label_files1 = sorted(glob.glob(os.path.join(data_path, "training\\1st_manual", "*.gif")))

image_files2 = sorted(glob.glob(os.path.join(data_path, "test\\images", "*.tif")))
label_files2 = sorted(glob.glob(os.path.join(data_path, "test\\1st_manual", "*.gif")))

image_files = image_files1 + image_files2
label_files = label_files1 + label_files2
'''

# 打乱文件列表
combined_files = list(zip(image_files, label_files))
random.shuffle(combined_files)
image_files, label_files = zip(*combined_files)

# 划分训练集和验证集（8:2）
split_idx = int(len(image_files) * 0.8)
dataconfig['train']['data'] = [
    {'image': img, 'label': lbl} 
    for img, lbl in zip(image_files[:split_idx], label_files[:split_idx])
]
dataconfig['val']['data'] = [
    {'image': img, 'label': lbl} 
    for img, lbl in zip(image_files[split_idx:], label_files[split_idx:])
]
dataconfig['test']['data'] = []

  
config['data'] = dataconfig

# ------------------- 配置文件检查 --------------------

check_config(config)

# ------------------- 写入训练配置 --------------------
configpath = 'config.json'
with open(configpath, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)
print(f"config file saved to: {os.path.abspath(configpath)}")
