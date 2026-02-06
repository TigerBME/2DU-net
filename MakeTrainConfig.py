import json
import os
import glob
import random
from typing import Dict, List, Tuple
from torch.cuda import is_available as cuda_available
from sklearn.model_selection import KFold
from Code2D.ConfigTool import *
from Code2D import model_train
from Code2D import get_default_config


def setup_logging_paths(config: Dict, fold_index: int) -> str:
    """更新配置中的日志路径并返回 log_path，每个fold使用不同的路径。"""
    log_path = getpath(f'TRAIN_RECORD_fold_{fold_index}', f'record_fold_{fold_index}')
    config['logging']['log_dir'] = os.path.join("./", log_path)
    config['early_stopping']['model_save_path'] = os.path.join(log_path, "best_model.pth")
    return log_path


def get_data_base_path(device: str) -> str:
    """根据设备返回数据根路径。"""
    if device == "cuda":
        return r"/root/WangDao/2DU-net/DATA/TrainData"
    else:
        return r"D:\C_File\WDdata\OCTA2D"


def build_image_label_dirs(data_path: str) -> Tuple[List[str], List[str]]:
    """返回 image_dirs 和 label_dirs 的列表（可包含多个子目录）。"""
    image_dirs = [os.path.join(data_path, "images"), os.path.join(data_path, "newimages2")]
    label_dirs = [os.path.join(data_path, "labels"), os.path.join(data_path, "newlabels2")]
    if len(image_dirs) != len(label_dirs):
        raise AssertionError("图像目录和标签目录数量不匹配！")
    return image_dirs, label_dirs


def gather_all_file_pairs(image_dirs: List[str], label_dirs: List[str]) -> List[Tuple[str, str]]:
    """收集所有 image_dirs 与 label_dirs 中所有匹配的图像-标签对。

    只查找扩展名为 .bmp 和 .png 的文件，按名称排序后配对。
    """
    all_pairs = []
    for img_dir, lbl_dir in zip(image_dirs, label_dirs):
        image_files: List[str] = []
        label_files: List[str] = []
        
        image_files.extend(sorted(glob.glob(os.path.join(img_dir, "*.bmp"))))
        image_files.extend(sorted(glob.glob(os.path.join(img_dir, "*.png"))))
        label_files.extend(sorted(glob.glob(os.path.join(lbl_dir, "*.bmp"))))
        label_files.extend(sorted(glob.glob(os.path.join(lbl_dir, "*.png"))))

        # 如果数量不匹配或为空，则跳过这对目录
        if not image_files or not label_files or len(image_files) != len(label_files):
            print(f"警告: 图像目录 {img_dir} 和标签目录 {lbl_dir} 中的文件数量不匹配或为空，跳过此对目录。")
            continue

        pairs = list(zip(image_files, label_files))
        all_pairs.extend(pairs)
    
    random.shuffle(all_pairs)
    return all_pairs


def populate_config_datasets_fold(config: Dict, train_indices: List[int], val_indices: List[int], all_pairs: List[Tuple[str, str]]) -> None:
    """清空并填充 config 中的 train/val 数据列表，根据给定的索引划分数据。"""
    config['data']['train']['data'] = []
    config['data']['val']['data'] = []
    config['data']['test']['data'] = []  # 测试集保持为空，或从训练集中划分
    
    # 填充训练集
    for idx in train_indices:
        if idx < len(all_pairs):
            image_path, label_path = all_pairs[idx]
            config['data']['train']['data'].append({'image': image_path, 'label': label_path})
    
    # 填充验证集
    for idx in val_indices:
        if idx < len(all_pairs):
            image_path, label_path = all_pairs[idx]
            config['data']['val']['data'].append({'image': image_path, 'label': label_path})


def save_config(config: Dict, configpath: str) -> str:
    """将配置写入文件并返回绝对路径。"""
    with open(configpath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    return os.path.abspath(configpath)


def make_k_fold_configs(k: int) -> List[str]:
    """"""
    print(f"执行 {k}-折交叉验证...")
    device = "cuda" if cuda_available() else "cpu"
    base_config = get_default_config(config_name="config_13.json")

    data_path = get_data_base_path(device)
    image_dirs, label_dirs = build_image_label_dirs(data_path)
    
    # 收集所有可用的数据对
    all_pairs = gather_all_file_pairs(image_dirs, label_dirs)

    print(f"一共 {len(all_pairs)} 组数据用于K折交叉验证。")
    
    # 初始化K折分割器
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # 获取所有数据的索引
    indices = list(range(len(all_pairs)))
    
    fold_configs = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(indices)):
        print(f"正在处理第 {fold_idx + 1}/{k} 折...")
        
        # 创建当前fold的配置副本
        fold_config = json.loads(json.dumps(base_config))  # 深拷贝配置
        
        # 设置当前fold的日志路径
        log_path = setup_logging_paths(fold_config, fold_idx)
        
        # 根据当前fold的索引划分数据
        populate_config_datasets_fold(fold_config, train_indices, val_indices, all_pairs)
        
        # 检查配置并写入 statistic 部分（如果 check_config 存在）
        try:
            check_config(fold_config)
        except NameError:
            print("注意: check_config 函数未定义，跳过配置检查。")
        
        # 保存当前fold的配置文件
        config_filename = f'config_fold_{fold_idx + 1}.json'
        saved_path = save_config(fold_config, config_filename)
        print(f"第 {fold_idx + 1} 折的配置文件已保存至: {saved_path}")

        fold_configs.append(saved_path)

    assert len(fold_configs) == k, "生成的配置文件数量与K值不匹配！"

    print("\n所有配置文件已生成:")
    for i, config_path in enumerate(fold_configs):
        print(f"  第 {i+1} 折: model_train('{config_path}')")

    return fold_configs


def run_train_config(configs_list: list[str], k: int, auto_run: bool = False) -> None:
    """根据用户输入决定是否运行K折交叉验证训练。

    如果 auto_run 为 True 则直接运行训练。
    """

    if auto_run:
        # 自动运行训练
        for i in range(k):
            config_path = configs_list[i]
            print(f"正在训练第 {i+1}/{k} 折,配置文件路径: {config_path}...")
            model_train(config_path)

        return

    print(f"是否执行 {k}-折交叉验证训练...")
    while True:
        user_input = input(f"是否执行 {k}-折交叉验证训练？(Y/N): ").strip().upper()
        if user_input == 'Y':
            # 执行依次训练
            for i in range(k):
                config_path = configs_list[i]
                print(f"正在训练第 {i+1}/{k} 折,配置文件路径: {config_path}...")
                model_train(config_path)
            break
        if user_input == 'N':
            # 不执行训练
            print(f"请手动运行训练代码...")
            break
        print("无效输入，请输入Y或N。")


def main(predict_now: bool = False, k: int = 5) -> None:
    print("开始制作用于K折交叉验证的数据...")
    
    configs = make_k_fold_configs(k)

    run_train_config(configs_list=configs, k=k, auto_run=predict_now)


if __name__ == '__main__':
    PREDICT_NOW = False
    K_FOLDS = 5  # 可以通过修改这个值来改变折数
    main(predict_now=PREDICT_NOW, k=K_FOLDS)
