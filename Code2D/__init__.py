# This code contains all we need 
# to train a 2D segmentation model
# But the data is not included 
# Use MakeTrainConfig.py to make new config 
# Then run ModelTrain.py by your config in the terminal
# The result will be saved in Record folder

import sys
import os

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录（假设当前文件在项目的一个子目录中）
# project_root = os.path.join(current_dir, '..')

# 将项目根目录添加到 sys.path
sys.path.append(current_dir)

import Model
import ConfigTool



# 导入modeltrain
# 导入modelpredict
from ModelTrain import main as model_train
from ModelPredict import main as model_predict

