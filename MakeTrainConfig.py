import json
import os
import glob
from torch.cuda import is_available as cuda_available
from Code2D.ConfigTool import *
from Code2D import model_train
from Code2D import get_default_config
import random


config = get_default_config(config_name="config_13.json")
# get_default_config()返回默认配置

# ------------------- 日志数据配置 -------------------
log_path = getpath('TRAIN_RECORD','record')
config['logging']['log_dir'] = os.path.join("./", log_path)
config['early_stopping']['model_save_path'] = os.path.join(log_path, "best_model.pth")
# ------------------- 训练数据配置 -------------------


# 动态生成数据路径
device = "cuda" if cuda_available() else "cpu"


# OCTA2D
print("Start making Data...")
if device == "cuda":
    # data_path = r"C:\\Users\\Dell\\Desktop\\WDprogram\\OCTA2D"
    data_path = r"/root/WangDao/2DU-net/DATA/TrainData"
else:
    data_path = r"D:\\C_File\\WDdata\\OCTA2D"


# 无卡开机时，直接指定数据路径
# data_path = r"D:\\C_File\\WDdata\\OCTA2D"
# data_path = r"/root/WangDao/Data"


# 定义四个子目录
image_dirs = [os.path.join(data_path, "images"),
              os.path.join(data_path, "newimages1"),
              os.path.join(data_path, "newimages2")
              ]

label_dirs = [os.path.join(data_path, "labels"),
                os.path.join(data_path, "newlabels1"),
                os.path.join(data_path, "newlabels2")
              ]

assert len(image_dirs) == len(label_dirs), "图像目录和标签目录数量不匹配！"

#  清空训练集、验证集、测试集
config['data']['train']['data'] = []
config['data']['val']['data'] = []
config['data']['test']['data'] = []

for i in range(len(image_dirs)):
    # 依次读取所有的目录
    image_files = []
    label_files = []
    image_files.extend(sorted(glob.glob(os.path.join(image_dirs[i], "*.bmp"))))
    label_files.extend(sorted(glob.glob(os.path.join(label_dirs[i], "*.png"))))

    # 打乱文件列表
    combined_files = list(zip(image_files, label_files))
    random.shuffle(combined_files)
    image_files, label_files = zip(*combined_files)

    split_idx = int(len(image_files) * 0.8)

    # 划分训练集和验证集（8:2）
    config['data']['train']['data'].extend(
        [
            {'image': img, 'label': lbl} 
            for img, lbl in zip(image_files[:split_idx], label_files[:split_idx])
        ]
    )
    config['data']['val']['data'].extend(
        [
            {'image': img, 'label': lbl} 
            for img, lbl in zip(image_files[split_idx:], label_files[split_idx:])
        ]
    )


# ------------------- 配置文件检查 --------------------

check_config(config) # 检查配置并写入statistic部分

# ------------------- 写入训练配置 --------------------
configpath = 'config.json'
with open(configpath, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)
print(f"config file saved to: \n{os.path.abspath(configpath)}")

# ------------------- 执行训练 -------------------
PREDICT_NOW = False
if PREDICT_NOW:
    model_train(configpath)
else:
    print(f"是否执行训练...")
    # 询问用户，输入Y执行，N退出，其余重新输入
    while True:
        user_input = input("是否执行训练？(Y/N): ").strip().upper()
        if user_input == 'Y':
            model_train(configpath)
            break
        elif user_input == 'N':
            print("退出程序,请手动执行训练代码")
            break
        else:
            print("无效输入，请输入Y或N。")