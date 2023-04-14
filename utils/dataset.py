import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import cv2
import os.path as osp
from itertools import chain
import json

_VALID_SPLITS = ('train', 'val', 'test')
_VALID_SCENE_TYPES = ('indoors', 'outdoor')


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_root, scene_types, splits, meta_fname, batch_size=6,
                 dim=(640, 480), n_channels=3, shuffle=True):
        """
        Initialization
        """
        self.data_root = data_root
        self.splits = self.check_and_tuplize_tokens(
            splits, _VALID_SPLITS
        )
        self.scene_types = self.check_and_tuplize_tokens(
            scene_types, _VALID_SCENE_TYPES
        )
        with open(meta_fname, 'r') as f:
            self.meta = json.load(f)

        imgs = []
        for split in self.splits:
            for scene_type in self.scene_types:
                _curr = self.enumerate_paths(self.meta[split][scene_type])
                _curr = map(lambda x: osp.join(self.data_root, split, scene_type, x), _curr)
                imgs.extend(list(_curr))
        self.imgs = imgs
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_depth = 0.1
        self.on_epoch_end()

    def check_and_tuplize_tokens(self, tokens, valid_tokens):
        if not isinstance(tokens, (tuple, list)):
            tokens = (tokens,)
        for split in tokens:
            assert split in valid_tokens
        return tokens

    def enumerate_paths(self, src):
        '''flatten out a nested dictionary into an iterable
        DIODE metadata is a nested dictionary;
        One could easily query a particular scene and scan, but sequentially
        enumerating files in a nested dictionary is troublesome. This function
        recursively traces out and aggregates the leaves of a tree.
        '''
        if isinstance(src, list):
            return src
        elif isinstance(src, dict):
            acc = []
            for k, v in src.items():
                _sub_paths = self.enumerate_paths(v)
                _sub_paths = list(map(lambda x: osp.join(k, x), _sub_paths))
                acc.append(_sub_paths)
            return list(chain.from_iterable(acc))
        else:
            raise ValueError('do not accept data type {}'.format(type(src)))

    def __len__(self):
        return int(np.ceil(len(self.imgs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        index_min = index * self.batch_size
        index_max = (index + 1) * self.batch_size

        list_ids = self.imgs[index_min:index_max]
        x, y = self.data_generation(list_ids)

        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle == True:
            np.random.shuffle(self.imgs)

    def load(self, image_path, depth_map_path, mask_path):
        image_ = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image_ = cv2.resize(image_, self.dim[::-1])
        image_ = tf.image.convert_image_dtype(image_, tf.float32)

        depth_map = np.load(depth_map_path).squeeze()

        mask = np.load(mask_path)
        mask = mask > 0

        max_depth = min(300, np.percentile(depth_map, 99))
        depth_map = np.clip(depth_map, self.min_depth, max_depth)
        depth_map = np.log(depth_map, where=mask)

        depth_map = np.ma.masked_where(~mask, depth_map)

        depth_map = np.clip(depth_map, 0.1, np.log(max_depth))
        depth_map = cv2.resize(depth_map, self.dim[::-1])
        depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
        depth_map = np.nan_to_num(depth_map, nan=0)
        depth_map = np.expand_dims(depth_map, axis=2)
        depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)
        return image_, depth_map

    def data_generation(self, list_ids):

        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        for i, path in enumerate(list_ids):
            image_path = path + '.png'
            depth_map_path = path + '_depth.npy'
            mask_path = path + '_depth_mask.npy'

            image_, depth_map = self.load(image_path, depth_map_path, mask_path)

            ### Data Augmentation
            x[i,] = image_
            y[i,] = depth_map

        return x, y
