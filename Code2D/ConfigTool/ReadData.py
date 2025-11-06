import os
import nibabel as nib
import numpy as np
from PIL import Image

def nifti_to_bmp(nifti_file_path: str, bmp_file_folder: str, cutting_dimension: int) -> list:
    '''
    从nifti文件中读取数据，沿指定维度(1，2，3)切片
    并保存为bmp文件。
    :param nifti_file_path: nifti文件路径
    :param bmp_file_folder: bmp文件保存路径
    :param cutting_dimension: 切片维度
    :return: 包含所有bmp文件地址的列表
    '''
    try:
        # 加载NIfTI文件
        nifti_image = nib.load(nifti_file_path)
        nifti_data = nifti_image.get_fdata()

        # 获取切片数量
        if cutting_dimension == 0:
            num_slices = nifti_data.shape[0]
        elif cutting_dimension == 1:
            num_slices = nifti_data.shape[1]
        elif cutting_dimension == 2:
            num_slices = nifti_data.shape[2]
        else:
            raise ValueError("切片维度应为0，1或2")
        

        # 确保保存文件夹存在
        if not os.path.exists(bmp_file_folder):
            os.makedirs(bmp_file_folder)

        bmp_paths = []

        # 遍历每个切片并保存为BMP文件
        for slice_index in range(num_slices):
            if cutting_dimension == 0:
                slice_data = nifti_data[slice_index, :, :]
            elif cutting_dimension == 1:
                slice_data = nifti_data[:, slice_index, :]
            elif cutting_dimension == 2:
                slice_data = nifti_data[:, :, slice_index]

            slice_data = slice_data.astype(np.uint8)

            # 创建图像并保存
            img = Image.fromarray(slice_data)
            img_path = os.path.join(bmp_file_folder, f'slice_{slice_index}.bmp')
            img.save(img_path)

            bmp_paths.append(img_path)

        return bmp_paths
    except Exception as e:
        print(f"转换过程中发生错误: {e}")
        return []
