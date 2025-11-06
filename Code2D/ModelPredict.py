from predict import predict_model
import json
import argparse
import torch
from torch.cuda import is_available as cuda_available
from torch import device as torch_device
from DataLoader import get_dataloader
from Model import create_model

def load_config(config_path):
    """加载配置文件"""
    with open(config_path) as f:
        return json.load(f)

def main(config_path):
    config = load_config(config_path)
    device = torch_device('cuda' if cuda_available() else 'cpu')

    data_config = config.pop('data')
    test_data_config = data_config.pop('test')
    test_config = config.pop('test')

    # 创建模型并加载权重
    try:
        model = create_model(config['model']).to(device)
        model.load_state_dict(torch.load(test_config['model_path'],map_location=device,weights_only=True))
        print(f"Loaded model from {test_config['model_path']}")
    except FileNotFoundError:
        print(f"Model not found at {test_config['model_path']}")
        print("Please make sure the model file exists and try again.")
        return

    # 获取测试数据加载器
    test_loader = get_dataloader(test_data_config, mode='test')
    print(f"Load {len(test_loader)} test data ")

    # 执行预测
    predict_model(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=test_config['output_dir'],
        threshold=test_config.get('threshold', 0.5)
    )
    print(f"Predict done, results saved to {test_config['output_dir']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    args = parser.parse_args()
    print(f"Loading config from {args.config}")
    main(args.config)
