# 该文件用于生成预测配置文件
# 使用时只需要修改 存档路径:record_folder_path 以及 输入数据位置:input_path
# 程序自动读取训练时使用的配置文件,并生成对应的预测配置文件

import json
import os
from Code2D.ConfigTool import getpath
from Code2D.ConfigTool.ReadData import nifti_to_png
from torch.cuda import is_available as cuda_available
import glob

# ------------------- 读取训练配置 -------------------
record_folder_path = r"D:\WangDao\2DTrainCode\Records\record29"

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
input_nifty_path = r"D:\WangDao\3DPreprocessCode\raw_data_nifty\10025\10025_nifty.nii.gz"
tempdir = getpath("Code2D","Temp") # 临时存放png文件的路径
input_files = nifti_to_png(input_nifty_path, tempdir, cutting_dimension=1) # 生成png文件并获取png文件路径列表

file_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(file_path, getpath("OUTPUTS","predict")) # 输出文件夹（也可另外指定）
os.makedirs(output_path, exist_ok=True) # 创建输出文件夹

input_type = "png" # 指定输入的图像数据类型

# 测试数据输入
# input_files = glob.glob(os.path.join(input_path,f"*.{input_type}"))

config['data']['test']['data'] = [
    {
        "image": file, 
        "name": os.path.join(output_path, os.path.basename(file))
    } 
    for file in input_files
] # 这里的name字段是预测结果的输出路径,默认与输入文件同名


# 模型输入
test_config = {
    "model_path":model_path,
    "output_dir":output_path,
}

config['test'] = test_config
# ------------------- 配置文件检查 -------------------
print(f"test picture: {len(config['data']['test']['data'])}")

# ------------------- 写回配置文件 -------------------
predict_config_path = os.path.join(record_folder_path, "predict_config.json")
with open(predict_config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)
print(f"配置文件已生成：\n{os.path.abspath(predict_config_path)}")
