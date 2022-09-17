import torch.utils.data as data
import cv2
import PIL.Image as Image
import pandas as pd
import os
import utils
import random
import numpy as np


class PdDataSet(data.Dataset):
    def __init__(self, data_path, label_path, label_name, mode='train', transform=None, basic_aug=False):
        """
        :param data_path: path of dataset
        :param mode: train or eval
        """
        super().__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.mode = mode
        self.transform = transform
        self.img_paths = []
        self.labels = []
        self.basic_aug = basic_aug
        self.aug_func = [utils.flip_image, utils.add_gaussian_noise]

        if mode == 'train':
            with open(os.path.join(self.label_path, label_name), "r", encoding="utf-8") as f:
                self.info = f.readlines()
        else:
            with open(os.path.join(self.label_path, label_name), "r", encoding="utf-8") as f:
                self.info = f.readlines()

        for img_info in self.info:
            img_path, label_pd = img_info.strip().split(' ')
            self.img_paths.append(os.path.join(self.data_path, img_path))
            self.labels.append(int(label_pd))


    def __getitem__(self, idx):
        """
        :param index: 
        :return:
        """
        # open the image and get the corresponding label
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        label = self.labels[idx]
        label = np.array([label], dtype="int64")

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def print_sample(self, index: int = 0):
        print("filename", self.img_paths[index], "\tlabel", self.labels[index])

    def __len__(self):
        return len(self.img_paths)
