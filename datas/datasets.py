# -*- coding: utf-8 -*-
import os
import re
from glob import glob
from PIL import Image

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class Datasets(object):
    r"""
    这个类为Dataloader的第一个参数
    data_dir_path 为数据集所在目标
    transforms为具体变化方法
    img_extension 为图像格式 默认为jpg
    """
    def __init__(self,data_dir_path,transforms,img_extension = 'jpg'):
        self.data_dir_path = data_dir_path
        self.transforms = transforms
        self.img_extension = img_extension
        # all_datas 为一个列表 存放数据集中所有数据的路径和标签
        # class_num 为数据集中类别总数
        self.all_datas,self.class_num = self.process()

    # process需要根据具体数据集重写
    def process(self,relabel=True):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        fpaths = sorted(glob(os.path.join(self.data_dir_path,'*.%s'%self.img_extension)))
        all_p_ids={}
        res = []
        for fpath in fpaths:
            fname = os.path.basename(fpath)
            p_id, cam_id = map(int,pattern.search(fname).groups())
            if p_id == -1:
                continue
            if relabel:
                if p_id not in all_p_ids:
                    all_p_ids[p_id] = len(all_p_ids)
            else:
                if p_id not in all_p_ids:
                    all_p_ids[p_id] = p_id
            p_id = all_p_ids[p_id]
            cam_id -=1
            res.append((fpath,p_id,cam_id))
        return res,int(len(all_p_ids))

    def __getitem__(self, item):
        fpath,p_id,cam_id = self.all_datas[item]
        with open(fpath,'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        # 在遍历dataloader时  这里返回几个 那边就有几个
        return img, p_id, cam_id,fpath

    def __len__(self):
        # 返回数据集的个数
        return len(self.all_datas)
