import os
import torch
import csv
from datetime import datetime

def getpath(foldername_root,recordname):
    '''
    输入文件夹名称,文件名称(任何名称不能含有括号)
    输出形如一下的路径：
    foldername/filename7
    (寻找最大的filename并加1)
    若没有该文件夹则自动创建
    
    //存在中间级文件夹时，只创建文件夹路径：(忽略filename参数)
    根文件夹/中间文件夹
    '''  
    # 创建文件夹
    if not os.path.exists(foldername_root):
        os.makedirs(foldername_root)

    name,ext = os.path.splitext(recordname)
    length = len(name)

    # 寻找最大序号
    files = os.listdir(foldername_root)
    series = []
    for file in files:
        file = file.split('(')[0] #去掉文件名尾部的括号
        filename,fileext = os.path.splitext(file)
        if filename.startswith(name) and fileext == ext:
            # 符合要求
            serial = filename[length:]
            try: 
                serial_number = int(serial)
                series.append(serial_number) 
            except ValueError:
                continue
        else:
            pass

    try:
        number = max(series) + 1
    except ValueError:
        number = 1

    path = os.path.join(foldername_root,name+str(number)+ext)
    
    if not ext:
        os.makedirs(path)
         
    return path

if __name__ =='__main__':
    pass


