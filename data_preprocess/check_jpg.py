# -*- coding: utf-8 -*-
"""
Created on 2023/10/13 14:24

@author: zhengjie
"""
import numpy as np
from PIL import Image

# 打开图像文件
# img = Image.open('G:\\rgb_chest\\8001848623_20200403_FILE145.jpg')
img = Image.open('G:\\rgb_chest\\8001850446_20200407_FILE5.jpg')

# 转换为 RGB 模式（如果不是的话）
# img = img.convert('RGB')

# 获取图像的像素数据
pixels = list(img.getdata())

# 打印前 10 个像素值
print(pixels[:5000])

