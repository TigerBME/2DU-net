from torch.utils.data import Dataset
from PIL import Image
import os

class PredictionDataset(Dataset):
    def __init__(self, data_list: list[dict], transform: callable=None, image_type: dict={'image': 'L'}):
        self.data_list = data_list
        self.transform = transform
        self.image_type = image_type['image']  # 保持字典结构访问

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path = self.data_list[idx]['image']        
        output_path = self.data_list[idx]['name']
        
        # 加载图像
        image = Image.open(image_path).convert(self.image_type)
        
        # 应用预处理
        if self.transform:
            image = self.transform(image)
            
        return image, output_path
