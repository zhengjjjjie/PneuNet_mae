# -*- coding: utf-8 -*-
"""
Created on 2023/10/17 17:20

@author: zhengjie
"""

import glob
import os
from PIL import Image
import os.path as osp

from tqdm import tqdm

"""
    解决 PIL.UnidentifiedImageError: cannot identify image file 问题
    过滤已经损坏的文件数据
"""

root = '/x32001107/rgb_chest/normal/'
dcm_files = glob.glob(osp.join(root, '**', '*.jpg'), recursive=True)
for f in tqdm(dcm_files):
    try:
        img = Image.open(f)
    except Exception as e:
        # 处理异常的代码
        if os.path.exists(f):
            print(f"{f} file is error and has removed it")
            os.remove(f)  # 删除文件
        print(f"异常是: {e}")
        continue
