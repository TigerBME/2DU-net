import json
import os
from Code2D.ConfigTool import getpath
import glob

# ------------------- 读取训练配置 -------------------

# record_folder_path = r"C:\Users\Dell\Desktop\WDprogram\Newcode\SpecialRecords\recordUnet400"
record_folder_path = r"C:\Users\Dell\Desktop\WDprogram\Newcode\Records\record25"
model_path = os.path.join(record_folder_path, "best_model.pth")
config_path = os.path.join(record_folder_path, "config.json")

config = dict()

# 读取config从config_path中读取config.json文件,并将其内容保存到config字典中
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# ------------------- 测试数据配置 -------------------
# 测试数据配置时只允许修改config['data']['test']['data']字段(即测试数据路径)
input_path = r"C:\Users\Dell\Desktop\WDprogram\3Dcode\Processed_data\OCTA\10010\Masked_pic"

file_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(file_path, getpath("OUTPUTS","predict"))
os.makedirs(output_path, exist_ok=True)

# 测试数据输入
input_files = glob.glob(os.path.join(input_path,"*.bmp"))
config['data']['test']['data'] = [
    {"image": file, "name": os.path.join(output_path, os.path.basename(file))} 
    for file in input_files
]

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
print(f"配置文件已生成：{os.path.abspath(predict_config_path)}")
