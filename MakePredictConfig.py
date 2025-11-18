# 该文件用于生成预测配置文件
# 使用时只需要修改 存档路径:record_folder_path 以及 输入数据位置:input_path
# 程序自动读取训练时使用的配置文件,并生成对应的预测配置文件

import json
import os
from Code2D.ConfigTool import getpath
import glob

# ------------------- 读取训练配置 -------------------
record_folder_path = r"D:\WangDao\2DTrainCode\Records\record24"

model_path = os.path.join(record_folder_path, "best_model.pth")
config_path = os.path.join(record_folder_path, "config.json")

config = dict()

# 读取config从config_path中读取config.json文件,并将其内容保存到config字典中
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# ------------------- 测试数据配置 -------------------
# 测试数据配置时只允许修改config['data']['test']['data']字段(即测试数据路径)
input_path = r"D:\WangDao\3DPreprocessCode\Processed_data\OCTA10010\Pictures"

file_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(file_path, getpath("OUTPUTS","predict"))

os.makedirs(output_path, exist_ok=True)

input_type = "png" # 指定输入的图像数据类型

# 测试数据输入
input_files = glob.glob(os.path.join(input_path,f"*.{input_type}"))
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
