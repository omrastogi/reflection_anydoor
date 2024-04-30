import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from data_utils import * 
from base import BaseDataset

class MirrorsCounterfactualDataset(BaseDataset):
    def __init__(self, data_dir):
        self.images_dir = f"{data_dir}/images"
        self.masks_dir = f"{data_dir}/masks"
        self.def_dir = f"{data_dir}/deformed"

        self.data = os.listdir(self.masks_dir)
        self.size = (512, 640)
        self.clip_size = (224,224)
        self.dynamic = 1

    def __len__(self):
        return len(os.listdir(self.masks_dir))

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H and w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H and w < W:
                pass_flag = False
        return pass_flag


    def get_sample(self, idx):
        ref_mask_path = os.path.join(self.masks_dir, self.data[idx])
        ref_image_path = os.path.join(self.def_dir, self.data[idx])
        tar_image_path = os.path.join(self.images_dir, self.data[idx])
        ref_mask = cv2.imread(ref_mask_path)
        ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2GRAY)

        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        tar_image = cv2.imread(tar_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

        # ref_mask = (ref_mask > 0).astype(np.uint8)
        tar_mask = ref_mask.copy()
        # cv2.imwrite('ref_mask0.png', ref_mask)

        item_with_collage = self.process_pairs_new(ref_image, ref_mask, tar_image, tar_mask, max_ratio = 1.0)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        return item_with_collage
    
if __name__ == "__main__":
    dataset = MirrorsDataset(data_dir="/raid/ankit/om/dataset/MSD/test")
    k = dataset[123]
    # print(k)
