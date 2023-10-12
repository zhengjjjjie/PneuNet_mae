import glob

import SimpleITK as sitk
import cv2
import os
import numpy as np


txt_list = 'train.txt'
with open(txt_list, 'r') as f:
    img_list = [line.strip() for line in f]
print("Processing {} datas".format(len(img_list)))

save_path = './nii2img/train'

im_list = []
skip = 0  # 采样间隔，可调整这个数字扩增数据量
n = 1  # 顺序存储图片
z_sum = 0

for idx in range(len(img_list)):
    # 获取路径
    ith_info = img_list[idx].split(" ")
    im_file = ith_info[0]
    mask_file = ith_info[1]
    pid = os.path.basename(im_file).split('_')[0]  # 病人的标识 id

    # 读取文件
    im = sitk.ReadImage(im_file)
    im = sitk.GetArrayFromImage(im)  # 转化为numpy
    mask = sitk.ReadImage(mask_file)
    mask = sitk.GetArrayFromImage(mask)

    if len(np.unique(mask)) > 2:  # 二分类问题，将大于 1的标签都设为 1
        mask[mask > 1] = 1

    print(f'im.shape: {im.shape}, mask.shape: {mask.shape}')  # shape: z, y, x
    print(f'unique mask: {np.unique(mask)}')  # 查看标签

    os.makedirs(save_path, exist_ok=True)

    for z in range(im.shape[0]):  # 读取每一层图像
        if np.max(mask[z, ...]) == 0 or np.count_nonzero(mask[z, ...]) < 40:
            # print('skip....')
            continue  # 没有标注的层直接忽略

        # 构建rgb三通道图像，包含当前层和上下层信息
        rgb_slice = np.zeros((im.shape[1], im.shape[2], 3))
        for k in range(3):
            idx = max(0, z + (k - 1) * skip)  # 防止下越界
            idx = min(im.shape[0] - 1, idx)  # 防止上越界
            # 在赋值前需要对灰度图像的值进行归一化，因为灰度值可能会超过 rgb的值，生成的 rgb可能会全白
            im_normalized = (im[idx, ...] - im[idx, ...].min()) / (im[idx, ...].max() - im[idx, ...].min()) * 255
            rgb_slice[:, :, k] = im_normalized

        mask_slice = mask[z, ...]

        cv2.imwrite(os.path.join(save_path, pid + '_' + str(n) + 'im.png'), rgb_slice)
        cv2.imwrite(os.path.join(save_path, pid + '_' + str(n) + 'mask.png'), np.uint8(mask_slice * 50))  # 这里乘以50是为了保存图像能看清

        # print(n)
        n = n+1

    z_sum = z_sum+im.shape[0]
print(n, z_sum)



# im = sitk.ReadImage('dataset_nii/test/07006_03.nii.gz')
# im = sitk.GetArrayFromImage(im)  # 转化为numpy
#
# mask = sitk.ReadImage('dataset_nii/test/07006_03_seg.nii.gz')
# mask = sitk.GetArrayFromImage(mask)
# print(im.shape, mask.shape)  # shape: z, y, x
# print(np.unique(mask))  # 查看标签
#
# save_path = 'wrong_nii'  # 用于保存png图像和mask图像
# os.makedirs(save_path, exist_ok=True)
#
# for z in range(im.shape[0]):  # 读取每一层图像
#     if np.max(mask[z, ...]) == 0:
#         print('skip....')
#         continue  # 没有标注的层直接忽略
#
#     # 构建rgb三通道图像，包含当前层和上下层信息
#     rgb_slice = np.zeros((im.shape[1], im.shape[2], 3))
#     skip = 1  # 采样间隔，可调整这个数字扩增数据量
#     for k in range(3):
#         idx = max(0, z + (k - 1) * skip)  # 防止下越界
#         idx = min(im.shape[0] - 1, idx)  # 防止上越界
#         # 在赋值前需要对灰度图像的值进行归一化，因为灰度值可能会超过 rgb的值，生成的 rgb可能会全白
#         im_normalized = (im[idx, ...] - im[idx, ...].min()) / (im[idx, ...].max() - im[idx, ...].min()) * 255
#         rgb_slice[:, :, k] = im_normalized
#
#     mask_slice = mask[z, ...]
#
#     cv2.imwrite(os.path.join(save_path, str(z + 1) + 'im.png'), rgb_slice)
#     cv2.imwrite(os.path.join(save_path, str(z + 1) + 'mask.png'), np.uint8(mask_slice * 50))  # 这里乘以50是为了保存图像能看清
