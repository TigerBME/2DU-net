from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, data_list: list[dict], 
                 data_transform:transforms.Compose, 
                 transform_targets: list[str],
                 picture_type: dict={'image': 'L', 'label': 'L'},
                 data_per_picture: int=1,
                 ):
        self.data_list = data_list
        self.data_transforms = data_transform.transforms
        self.transform_targets = transform_targets
        self.imagetype = picture_type['image']
        self.labeltype = picture_type['label']
        
        self.data_per_picture = data_per_picture
        # 非数据增强情况下为1，数据增强情况下为增强倍数

        assert len(self.transform_targets) == len(self.data_transforms), \
            "transform_targets and data_transform must have the same length"

    def __len__(self):
        return len(self.data_list) * self.data_per_picture  # 修正乘法运算

    def __getitem__(self, data_idx):
        idx = data_idx // self.data_per_picture
        item = self.data_list[idx]
        image_path = item['image']
        label_path = item['label']

        # 加载图像和标签
        image = Image.open(image_path).convert(self.imagetype)
        label = Image.open(label_path).convert(self.labeltype)

        # 应用预处理
        if self.data_transforms is not None:
            for transform, target in \
                zip(self.data_transforms, self.transform_targets):

                if target == 'image':
                    image = transform(image)
                elif target == 'label':
                    label = transform(label)
                elif target == 'both':
                    image, label = transform(image, label)
                else:
                    raise ValueError(f"Invalid transform target: {target}")

        return image, label
