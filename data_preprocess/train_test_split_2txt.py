import glob
import os
import os.path as osp

import nibabel
import numpy as np
from sklearn.model_selection import train_test_split

root_dir = 'F:/pneumonia'
print("Processing datas from {0}".format(root_dir))

im_list = []

# 获取所有的.nii.gz文件路径，包括其所有子文件夹
nii_files = glob.glob(osp.join(root_dir, '**', '*.nii.gz'), recursive=True)
for nii_file in nii_files:
    if 'mask' in nii_file:
        im_file = nii_file.replace('_mask', '')  # 图片
        mask_file = nii_file  # 掩码
        pid = osp.basename(im_file).split('_')[0]  # 病人的标识 id
        clinic_file = osp.join(osp.dirname(im_file), pid + '_clinics.xlsx')
        ct_observ_file = osp.join(osp.dirname(im_file), pid + '_ct_obervations.xlsx')
        info = (im_file, mask_file, clinic_file, ct_observ_file)

        if os.path.getsize(im_file) == 0 or os.path.getsize(mask_file) == 0:
            print("{0} File is empty.".format(im_file))
            continue

        mask = nibabel.load(mask_file).get_fdata()
        mask_array = np.array(mask)
        if np.max(mask_array) == 0 or np.count_nonzero(mask_array) < 100:
            print("{0} mask File is incompatible.".format(im_file))
            continue
        
        # 过滤label>1的数据
        if np.max(mask_array) > 1:
            print("{0} mask File's label is error.".format(im_file))
            continue

        im_list.append(info)
print('Total samples:', len(im_list))
# i = 1
# for info in im_list:
#     print(i)
#     i = i+1
#     print(info[2])

# 划分数据集
train_size = 0.8  # 80% 作为训练集
train_list, test_list = train_test_split(im_list, train_size=train_size, random_state=42)

# 保存训练集路径信息到 train.txt
with open("train.txt", "w") as train_file:
    for info in train_list:
        train_file.write(" ".join(info) + "\n")

# 保存测试集路径信息到 test.txt
with open("test.txt", "w") as test_file:
    for info in test_list:
        test_file.write(" ".join(info) + "\n")

print("Train samples:", len(train_list))
print("Test samples:", len(test_list))
