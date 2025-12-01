from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch


class UnlabeledDataset(Dataset):
    """
    该数据集仅加载图像，不加载真实标签。
    为了保持原有训练代码结构，会返回一个与图像同尺寸的全黑标签图像。
    """
    def __init__(self, 
                 data_list: list[dict],
                 data_transform: transforms.Compose,
                 transform_targets: list[str],
                 picture_type: dict = {'image': 'L', 'label': 'L'},
                 data_per_picture: int = 1):

        self.data_list = data_list
        self.data_transforms = data_transform.transforms
        self.transform_targets = transform_targets

        self.imagetype = picture_type['image']
        self.labeltype = picture_type['label']

        self.data_per_picture = data_per_picture

        assert len(self.transform_targets) == len(self.data_transforms), \
            "transform_targets and data_transform must have the same length"

    def __len__(self):
        return len(self.data_list) * self.data_per_picture

    def __getitem__(self, data_idx):
        idx = data_idx // self.data_per_picture
        item = self.data_list[idx]

        image_path = item['image']


        # -------- 1. 加载图像 --------
        image = Image.open(image_path).convert(self.imagetype)

        # -------- 2. 加载标签标签（尺寸一致） --------
        # 如果有标签，加载之
        if 'label' in item:
            label_path = item['label']
            label = Image.open(label_path).convert(self.labeltype)
        else:
            # 如果没有标签，生成全黑标签
            label = Image.new(self.labeltype, image.size, 0)

        # -------- 3. 应用与原版完全相同的 transform 规则 --------
        if self.data_transforms is not None:
            for transform, target in zip(self.data_transforms, self.transform_targets):
                if target == 'image':
                    image = transform(image)
                elif target == 'label':
                    # 虽然不会使用 label，但 transform 过程不变
                    label = transform(label)
                elif target == 'both':
                    image, label = transform(image, label)
                else:
                    raise ValueError(f"Invalid transform target: {target}")

        return image, label
