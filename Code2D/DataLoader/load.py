from torch.utils.data import DataLoader
from PIL import Image
import sys
import torchvision.transforms as transforms
from . import Custom_transforms
from .TrainDataset import CustomDataset
from .PredictionDataset import PredictionDataset
from .SelfTrainDataset import UnlabeledDataset

current_module = sys.modules[__name__]

def build_transforms(transform_config:list[dict]):
    """
    根据配置动态构建预处理流程
    解析参数得到transforms.Compose对象与对应的操作对象列表
    
    """
    transform_list = []
    target_list = []
    for t in transform_config:
        transform_type = t.get('type')
        params = t.get('params', {})
        target = t.get('target')

        try:
            transform = getattr(Custom_transforms, transform_type, None)
            if transform is not None:
                # 该预处理方法为自编写
                transform = transform(**params)
                transform_list.append(transform)
                target_list.append(target)
                # 直接加入list

            else:
                # 没有找到，搜索torchvision预处理方法
                transform = getattr(transforms, transform_type)
                # 该预处理方法为torchvision自带
                if target == 'both':
                    # 需要同时对图像和标签进行预处理
                    # print(f"The {transform_type} transform is for both image and label")
                    transform = transform(**params)
                    transform_list.append(transform)
                    transform_list.append(transform)
                    target_list.append('image')
                    target_list.append('label')
                else:
                    # 只对图像进行预处理
                    transform = transform(**params)
                    transform_list.append(transform)
                    target_list.append(target)

        except AttributeError:
            raise ValueError(f"Invalid transform type: {transform_type}")
        
    return transforms.Compose(transform_list), target_list

def get_dataloader(data_config:dict, mode:str):
    '''
    data_config: test_config/train_config/val_config
    mode: 'test'/'train'/'val'
    '''
    if mode == 'test':
        # 测试数据预处理
        data_transforms, transforms_targets = build_transforms(data_config['preprocess'])
        
        if "label" in transforms_targets:
            # 测试集不应当出现以标签为目标的预处理
            raise ValueError("Test dataset should not have label transform")

        # 测试数据集
        test_dataset = PredictionDataset(
            data_list=data_config['data'],
            transform=data_transforms,
            image_type=data_config['picture_type']
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            **data_config['dataloader_args']
        )
        return test_dataloader
    
    elif mode == 'train' or mode == 'val':
        # 训练/验证数据预处理
        data_transforms, transforms_targets = build_transforms(data_config['preprocess'])
        # image_transforms = build_transforms(data_config['preprocess']['image'])
        # label_transforms = build_transforms(data_config['preprocess']['label'])
        
        # 初始化数据集
        Dataset = getattr(current_module, data_config['dataset_type'])

        dataset = Dataset(
            data_list=data_config['data'],
            data_transform=data_transforms,
            transform_targets=transforms_targets,
            picture_type=data_config['picture_type'],
            data_per_picture=data_config['data_per_picture']
        )

        # 创建DataLoader
        loader = DataLoader(
            dataset,
            **data_config['dataloader_args']
        )
        return loader
    
    else:
        raise ValueError(f"Invalid mode: {mode}")
