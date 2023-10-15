import SimpleITK as sitk
import os
import os.path as osp
import numpy as np
import glob
import cv2


def hu2gray(volume, WL=-500, WW=1200):
    """
    convert HU value to gray scale[0,255] using lung-window(WL/WW=-500/1200)
    :param volume:图像
    :param WL:窗位
    :param WW:窗宽
    :return:
    """
    low = WL - 0.5 * WW
    volume = (volume - low) / WW * 255.0
    volume[volume > 255] = 255
    volume[volume < 0] = 0
    volume = np.uint8(volume)
    return volume


src_root = 'G:\\pneumonia_data\\SWH_normal_person'  # normal person
dst_root = 'G:\\rgb_chest'  # rgb_chest
os.makedirs(dst_root, exist_ok=True)

dcm_files = glob.glob(osp.join(src_root, '**', '*.dcm'), recursive=True)
total = len(dcm_files)
done = 0
for f in dcm_files:
    print(done, total, f)
    done += 1

    try:
        im = sitk.GetArrayFromImage(sitk.ReadImage(f))
        lung = hu2gray(im.copy(), WL=-500, WW=1200).transpose((1, 2, 0))  # 肺窗图像
        mediastinum = hu2gray(im.copy(), WL=40, WW=400).transpose((1, 2, 0))  # 纵膈窗图像
        bone = hu2gray(im.copy(), WL=300, WW=1500).transpose((1, 2, 0))  # 骨窗图像
    except BaseException as e:
        print(f"{f} file caused an error occurred: {e}")
        continue

    merge = np.concatenate((lung, mediastinum, bone), axis=-1)  # 将三种图像拼接成rgb图像

    merge = cv2.resize(merge, (256, 256))  # 将图像缩小一倍，降低空间和模型计算量，以后模型输入大小就是256x256
    if np.count_nonzero(merge > 100) < 200 or len(np.unique(bone)) < 3:  #
        # 过滤掉目标太小的图像和全红全黑的数据
        print(f"{f} file is incompatible")
        continue
    if np.count_nonzero(lung > 10) < 12000 or np.count_nonzero(lung > 10) < 12000 or np.count_nonzero(
            mediastinum > 10) < 12000:  # 过滤非黑区域小于20%的数据
        print(f"{f} file is error")
        continue
    save_name = osp.basename(osp.dirname(f)) + '_' + osp.basename(f)
    save_name = save_name.replace('.dcm', '.jpg')
    merge = np.flip(merge, axis=-1)
    cv2.imwrite(osp.join(dst_root, save_name), merge)
