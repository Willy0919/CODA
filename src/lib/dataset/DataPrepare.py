#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Willy

import sys
import os
from PIL import Image


def prepareShangHaiTech():
    DATASET_ROOT = 'data/ShanghaiTech'
    for part in ['part_A_final','part_B_final']:
        for phase in ['train_data','test_data']:
            DATASET_PATH = os.path.join(DATASET_ROOT,part,phase)
            fout = open(DATASET_PATH+'.txt','w+')
            for img_name in os.listdir(os.path.join(DATASET_PATH, 'images')):
                image_path = os.path.join(DATASET_PATH, 'images', img_name)
                gt_path = os.path.join(DATASET_PATH, 'ground_truth', 'GT_' + img_name.split('.')[0] + '.mat')
                fout.write(image_path + ' ' + gt_path + '\n')
            fout.close()

nameFuncMapping = {'shanghai':prepareShangHaiTech}

#def Mirror()

def main(argv):
    nameFuncMapping[argv[0]]()
    return 0

if __name__ == '__main__':
    main(sys.argv[1:])
