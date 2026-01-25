# 该文件用于生成预测配置文件
# 使用时只需要修改 存档路径:record_folder_path 以及 输入数据位置:input_path
# 程序自动读取训练时使用的配置文件,并生成对应的预测配置文件

import json
import os
from Code2D.ConfigTool import getpath
from Code2D.ConfigTool.ReadData import nifti_to_png
from Code2D import model_predict
from torch.cuda import is_available as cuda_available
import glob


# ------------------- 读取训练配置 -------------------
record_folder_path = r"/root/WangDao/Record/record13"

model_path = os.path.join(record_folder_path, "best_model.pth")
config_path = os.path.join(record_folder_path, "config.json")

config = dict()

# 读取config从config_path中读取config.json文件,并将其内容保存到config字典中
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# ------------------- 测试数据配置 -------------------
# 测试数据配置时只允许修改config['data']['test']['data']字段(即测试数据路径)
# if cuda_available():
#     input_path = r"/root/WangDao/Data/Pictures"
# else:
#     input_path = r"D:\WangDao\3DPreprocessCode\Processed_data\OCTA10270\Pictures"

# 输入的nifty文件路径
input_nifty_path = r"/root/WangDao/DATA/10025/Masked_volume.nii.gz"
tempdir = getpath("Code2D","Temp") # 临时存放png文件的路径
input_files = nifti_to_png(input_nifty_path, tempdir, cutting_dimension=1) # 生成png文件并获取png文件路径列表

file_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(file_path, getpath("PREDICT_OUTPUTS","predict")) # 输出文件夹（也可另外指定）
os.makedirs(output_path, exist_ok=True) # 创建输出文件夹
os.makedirs(os.path.join(output_path, "pictures"), exist_ok=True) # 创建存放预测图片的文件夹

input_type = "png" # 指定输入的图像数据类型

# 测试数据输入
# input_files = glob.glob(os.path.join(input_path,f"*.{input_type}"))

config['data']['test']['data'] = [
    {
        "image": file, 
        "name": os.path.join(output_path, "pictures", os.path.basename(file))
    } 
    for file in input_files
] # 这里的name字段是预测结果的输出路径,默认与输入文件同名


# 模型输入
test_config = {
    "model_path":model_path,
    "output_dir":output_path,
    "binarization":{
        "name": "HysteresisThreshold",
        "low_threshold": 0.4,
        "high_threshold": 0.8
    },
    "input_nifty": input_nifty_path,
}

config['test'] = test_config
# ------------------- 配置文件检查 -------------------
print(f"test picture: {len(config['data']['test']['data'])}")

# ------------------- 写回配置文件 -------------------
predict_config_path = os.path.join(record_folder_path, "predict_config.json")
with open(predict_config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)
print(f"配置文件已生成：\n{os.path.abspath(predict_config_path)}")

# ------------------- 执行预测 -------------------
PREDICT_NOW = True
if PREDICT_NOW:
    model_predict(predict_config_path)
else:
    print(f"是否执行预测...")
    # 询问用户，输入Y执行，N退出，其余重新输入
    while True:
        user_input = input("是否执行预测？(Y/N): ").strip().upper()
        if user_input == 'Y':
            model_predict(predict_config_path)
            break
        elif user_input == 'N':
            print("退出程序,请手动执行预测代码")
            break
        else:
            print("无效输入，请输入Y或N。")