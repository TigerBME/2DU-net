import os
import nibabel as nib
import numpy as np
from PIL import Image

def nifti_to_bmps(nifti_file: nib.Nifti1Image, cutting_dimension: int = 1) -> list[Image.Image]:
    '''
    将NIfTI图像数据沿指定维度切片并转换为PIL图像列表
    :param nifti_file: 输入的NIfTI图像文件
    :param cutting_dimension: 指定沿哪个维度进行切片（默认为1）
    :return: PIL图像列表
    '''
    data = nifti_file.get_fdata()
    num_slices = data.shape[cutting_dimension]
    images = []
    
    for slice_index in range(num_slices):
        # 使用numpy的索引功能，将任意维度的切片提取出来
        slice_data = np.take(data, indices=slice_index, axis=cutting_dimension)
        # 数据归一化并转换为uint8
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data) + 1e-8) * 255
        slice_data = slice_data.astype(np.uint8)
        img = Image.fromarray(slice_data)
        images.append(img)
    return images

def bmps_to_nifti(pictures: list[Image.Image], stacking_dimension: int = 2) -> nib.Nifti1Image:
    '''
    将PIL图像列表转换为三维NIfTI图像，并沿指定维度堆叠
    :param pictures: 输入的PIL图像列表
    :param stacking_dimension: 指定沿哪个维度进行堆叠（默认为2）
    :return: 三维NIfTI图像对象
    '''
    # 获取图像尺寸并初始化三维数组
    width, height = pictures[0].size
    num_slices = len(pictures)
    if stacking_dimension == 0:
        data = np.zeros((num_slices, height, width), dtype=np.uint8)
    elif stacking_dimension == 1:
        data = np.zeros((height, num_slices, width), dtype=np.uint8)
    else:
        data = np.zeros((height, width, num_slices), dtype=np.uint8)
    
    for i, img in enumerate(pictures):
        img_array = np.array(img)
        # 沿指定维度赋值
        if stacking_dimension == 0:
            data[i, :, :] = img_array
        elif stacking_dimension == 1:
            data[:, i, :] = img_array
        else:
            data[:, :, i] = img_array
    
    # 创建NIfTI对象（使用默认affine矩阵）
    affine = np.eye(4)
    return nib.Nifti1Image(data, affine)

def save_nifti(nifti_file: nib.Nifti1Image, save_path: str) -> None:
    '''
    保存NIfTI文件到指定路径
    '''
    nib.save(nifti_file, save_path)

def load_nifti(nifti_path: str) -> nib.Nifti1Image:
    '''
    从路径加载NIfTI文件
    '''
    return nib.load(nifti_path)

def save_bmps(pictures: list[Image.Image], save_path: str) -> None:
    '''
    保存PIL图像列表到指定文件夹
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for i, img in enumerate(pictures):
        img.save(os.path.join(save_path, f'slice_{i}.bmp'))

def load_bmps(bmps_path: str) -> list[Image.Image]:
    '''
    从文件夹加载所有BMP图像并排序
    '''
    files = [f for f in os.listdir(bmps_path) if f.endswith('.bmp')]
    # 按文件名中的数字序号排序
    files.sort()
    
    images = []
    for file in files:
        img = Image.open(os.path.join(bmps_path, file))
        images.append(img)
    return images


if __name__ == '__main__':
    nii_path = "mask.nii.gz"  # 掩膜文件路径
    pictures_folder = "bmp_slices"  # BMP切片文件夹路径

    # 加载BMP切片并转换为NIfTI
    pictures = load_bmps(pictures_folder)
    seg_nii = bmps_to_nifti(pictures)

    # 加载掩膜NIfTI文件
    masknii = load_nifti(nii_path)
    
    # 对segnii应用掩膜（直接相乘）
    seg_data = seg_nii.get_fdata().astype(np.float32)  # 转换为浮点避免溢出
    mask_data = masknii.get_fdata()
    
    # 维度校验
    if seg_data.shape != mask_data.shape:
        raise ValueError("Error: seg和mask的维度不匹配")
    
    # 应用掩膜并恢复数据类型
    seg_masked_data = (seg_data * mask_data).astype(np.uint8)
    
    # 更新NIfTI数据
    seg_nii = nib.Nifti1Image(seg_masked_data, affine=seg_nii.affine)

    # 保存结果
    save_nifti(seg_nii, "seg_result.nii.gz")

    