import copy
import random
import logging
import json

import cv2
import numpy
import torch
import numpy as np
import os
import utils

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class All_Dataset(Dataset):
    def __init__(self, cfg, root, is_train, datatype, transform=None):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.is_train = is_train
        self.root = root

        self.Fraction = cfg.ALL.FRACTION
        self.Data_Format = cfg.ALL.DATA_FORMAT
        self.Transform = transform

        if datatype == 'WFLW':
            annotation_file_WFLW = os.path.join(root, "WFLW", 'WFLW_annotations', 'list_98pt_rect_attr_train_test',
                                                    'list_98pt_rect_attr_test.txt')
            Database_WFLW = self.get_file_information_WFLW(annotation_file_WFLW)
            self.database = Database_WFLW
        elif datatype == '300W':
            annotation_file_300W = os.path.join(root, '300W', 'test_list.txt')
            Database_300W = self.get_file_information_300W(annotation_file_300W)
            self.database = Database_300W
        elif datatype == 'COFW':
            annotation_file_COFW = os.path.join(root, "COFW", 'test_list.txt')
            Database_COFW = self.get_file_information_COFW(annotation_file_COFW)
            self.database = Database_COFW
        else:
            raise NotImplementedError


    def get_file_information_WFLW(self, annotation_file):
        Data_base = []

        with open(annotation_file) as f:
            info_list = f.read().splitlines()
            f.close()

        for temp_info in info_list:
            temp_point = []
            temp_info = temp_info.split(' ')
            for i in range(2 * 98):
                temp_point.append(float(temp_info[i]))
            point_coord = np.array(temp_point, dtype=np.float).reshape(98, 2)
            max_index = np.max(point_coord, axis=0)
            min_index = np.min(point_coord, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0],
                                 max_index[1] - min_index[1]])
            temp_name = os.path.join(self.root, "WFLW", 'WFLW_images', temp_info[-1])
            Data_base.append({'Img': temp_name,
                              'bbox': temp_box,
                              'point': point_coord})

        return Data_base

    def get_file_information_300W(self, annotation_file):
        Data_base = []

        with open(annotation_file) as f:
            info_list = f.read().splitlines()
            f.close()

        for temp_info in info_list:
            temp_name = os.path.join(self.root, "300W", temp_info)
            Points = np.genfromtxt(temp_name[:-3] + 'pts', skip_header=3, skip_footer=1, delimiter=' ') - 1.0

            max_index = np.max(Points, axis=0)
            min_index = np.min(Points, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0], max_index[1] - min_index[1]])
            Data_base.append({'Img': temp_name,
                              'bbox': temp_box,
                              'point': Points,
                              })

        return Data_base

    def get_file_information_COFW(self, annotation_file):
        Data_base = []

        with open(annotation_file) as f:
            info_list = f.read().splitlines()
            f.close()

        for temp_info in info_list:

            temp_name = os.path.join(self.root, "COFW", temp_info) + '.jpg'

            Points = np.load(temp_name[:-4] + '.npz')['Points']

            max_index = np.max(Points, axis=0)
            min_index = np.min(Points, axis=0)
            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0], max_index[1] - min_index[1]])

            Data_base.append({'Img': temp_name,
                              'bbox': temp_box,
                              'point': Points})

        return Data_base


    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])

        # 读取信息
        Img_path = db_slic['Img']
        BBox = db_slic['bbox']
        Points = db_slic['point']

        initial_points = Points.copy()
        Img = cv2.imread(Img_path)

        Img_shape = Img.shape
        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)

        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)

        if len(Points) == 29:
            Scale = self.Fraction * 1.1
        else:
            Scale = self.Fraction

        trans = utils.get_transforms(BBox, Scale, 0.0, self.Image_size, shift_factor=[0.0, 0.0])

        input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

        for i in range(len(Points)):
            Points[i, 0:2] = utils.affine_transform(Points[i, 0:2], trans)

        meta = {
                'initial': initial_points,
                'Img_path': Img_path,
                'Points': Points / (self.Image_size),
                'trans': trans,
            }

        if self.Transform is not None:
            input = self.Transform(input)

        return input, meta


