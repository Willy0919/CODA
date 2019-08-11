#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Willy

from skimage import io,color
from skimage import transform as sk_transform
from src.lib.network.cn import vgg16
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy.io as scio
import os
from src.lib.utils.image_opt import findSigma,genDensity
warnings.filterwarnings("ignore")


def imread(path):
    image = io.imread(path)
    if len(image.shape) == 2:
        image = color.gray2rgb(image)
    return image

def preferred_resize(image,size):
    h,w,c = image.shape
    ratio  = 1.0
    if h > w:
        new_h, new_w = size, int(size * w / h + 0.5)
        ratio = size/h
    else:
        new_h, new_w = int(size * h / w + 0.5), size
        ratio = size/w
    resized_image = sk_transform.resize(image, (new_h, new_w), preserve_range=True)
    out_image = np.zeros((size, size, c), dtype=np.float32)
    out_image[...] = 127.5
    out_image[:new_h, :new_w, :] = resized_image

    return out_image,ratio

def ToTensor(image):
    image = image.transpose((2, 0, 1))
    return image

def normalize(image):
    image = (image - 127.5) / 127.5
    return image

def loadModel( path, branch):
    model = branch(pretrained=False)
    model.create_architecture()
    #print(torch.load(path))
    model.load_state_dict(torch.load(path))
    model.train(False)
    return model




def showTest(image, output,gt):
    image = (image * 127.5 + 127.5).astype(np.uint8)
    image = image.transpose((1, 2, 0))
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set_title('original image', fontsize=14)
    ax.imshow(image)
    ax = fig.add_subplot(223)
    ax.set_title('gt count: {:.2f}'.format(np.sum(gt)))
    ax.imshow(gt, cmap='jet', interpolation='bilinear')

    ax = fig.add_subplot(224)
    ax.set_title('predict count: {:.2f}'.format(np.sum(output)))
    ax.imshow(output[0, 0, :], cmap='jet', interpolation='bilinear')
    plt.show()


def density(image,path,ratio=1):
    gt = scio.loadmat(path, struct_as_record=False, squeeze_me=True)
    info = gt['image_info']
    dots = info.location
    print(len(dots))
    dots = dots*ratio
    #sigma = findSigma(dots, 2, 0.3)
    sigma = torch.FloatTensor(len(dots)).fill_(15)
    densityMap = genDensity(image, dots, sigma, 1001)
    print(np.sum(densityMap))
    if np.sum(densityMap) != 0:
        densityMap = densityMap * (len(dots) / np.sum(densityMap))
    return densityMap,len(dots)



def test(use_gpu, model_path, image_list):
    model = loadModel(model_path, vgg16)

    for path in image_list:
        print(path)
        path = os.path.join('/home/wl/PycharmProjects/0321/CSRNet_shanghaitechb/data/ShanghaiTech/part_B_final/test_data/images',path)
        raw = imread(path)

        image = normalize(ToTensor(raw))
        gt_path = path.replace('images', 'ground_truth').replace('IMG', 'GT_IMG').replace('.jpg', '.mat')
        dmap,gt_count = density(raw,gt_path)

        if use_gpu:
            torchimage = Variable(torch.FloatTensor(image).unsqueeze(dim=0).cuda())

            model = model.cuda()
        else:
            torchimage = Variable(torch.FloatTensor(image).unsqueeze(dim=0))
        #print('image type:', torchimage)
        densityMap,feature_map = model(torchimage)
        output = densityMap.data.cpu().numpy()
        #print(output)
        #print(dmap)
        count = np.sum(output.reshape((output.shape[0], -1)), 1)
        print('predict count:',count)
        print('gt count:', np.sum(dmap),gt_count)
        if gt_count>200 and abs(gt_count - count) <= 5:
            showTest(image,output,dmap)

def get_feature_elsewise(feature):
    c,h,w = feature.shape
    elsewise_mat = np.zeros((h,w))
    for i in range(c):
        elsewise_mat = elsewise_mat + feature[i,:,:]

    return elsewise_mat

def data_analysis(use_gpu, model_path, image_list,resize=0):
    model = loadModel(model_path, vgg16)
    running_mse = 0.0
    running_mae = 0.0

    for path in image_list:
        print(path)

        raw = imread(path)
        if resize:
            resize_img,ratio = preferred_resize(raw,resize)
        else:
            resize_img = raw
            ratio=1

        image = normalize(ToTensor(resize_img))
        gt_path = path.replace('images', 'ground_truth').replace('IMG', 'GT_IMG').replace('.jpg', '.mat')
        dmap, gt_count = density(resize_img, gt_path,ratio)

        if use_gpu:
            torchimage = Variable(torch.FloatTensor(image).unsqueeze(dim=0).cuda())

            model = model.cuda()
        else:
            torchimage = Variable(torch.FloatTensor(image).unsqueeze(dim=0))
        # print('image type:', torchimage)
        densityMap = model(torchimage)
        output = densityMap.data.cpu().squeeze().numpy()
        print('gt count:{} predict count:{:.4f}'.format(gt_count, np.sum(output)))
        running_mae += np.sum(np.abs(np.sum(output) - gt_count))
        running_mse += np.sum((np.sum(output) - gt_count) ** 2)

        # if gt_count > 200 and abs(gt_count - np.sum(output)) <= 5:
        #     #showTest(image, output, dmap)
        #
        #     fig = plt.figure()
        #     image_show = (image * 127.5 + 127.5).astype(np.uint8)
        #     image_show = image_show.transpose((1, 2, 0))
        #     ax = fig.add_subplot(231)
        #     ax.set_title('original image', fontsize=14)
        #     ax.imshow(image_show)
        #     ax = fig.add_subplot(232)
        #     ax.set_title('gt count: {:.2f}'.format(np.sum(dmap)))
        #     ax.imshow(dmap, cmap='jet', interpolation='bilinear')
        #     ax = fig.add_subplot(233)
        #     ax.set_title('predict count: {:.2f}'.format(np.sum(output)))
        #     ax.imshow(output, cmap='jet', interpolation='bilinear')
        #     # ax = fig.add_subplot(334)
        #     # ax.set_title('feature map 3')
        #     # ax.imshow(get_feature_elsewise(feature_3), cmap='jet', interpolation='bilinear')
        #     #
        #     # ax = fig.add_subplot(335)
        #     # ax.set_title('feature map 8')
        #     # ax.imshow(get_feature_elsewise(feature_8), cmap='jet', interpolation='bilinear')
        #     #
        #     # ax = fig.add_subplot(336)
        #     # ax.set_title('feature map 15')
        #     # ax.imshow(get_feature_elsewise(feature_15), cmap='jet', interpolation='bilinear')
        #     ax = fig.add_subplot(234)
        #     ax.set_title('feature map 15')
        #     ax.imshow(get_feature_elsewise(feature_15), cmap='jet', interpolation='bilinear')
        #     ax = fig.add_subplot(235)
        #     ax.set_title('feature map 22')
        #     ax.imshow(get_feature_elsewise(feature_22), cmap='jet', interpolation='bilinear')
        #     ax = fig.add_subplot(236)
        #     ax.set_title('feature map 29')
        #     ax.imshow(get_feature_elsewise(feature_29), cmap='jet', interpolation='bilinear')
        #
        #     plt.show()



        # print(output)
        # print(dmap)
        # count = np.sum(output.reshape((output.shape[0], -1)), 1)
        # print('predict count:', count)
        # print('gt count:', np.sum(dmap), gt_count)
        # if gt_count > 200 and abs(gt_count - count) <= 5:
        #     showTest(image, output, dmap)

    epoch_mae = running_mae/ len(image_list)
    epoch_mse = np.sqrt(running_mse / len(image_list))
    print('MAE:{:.4f} MSE:{:.4f}'.format(epoch_mae, epoch_mse))


if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    model_path = 'tools/parta.model'
    image_path = 'data/ShanghaiTech/part_B_final/test_data/images'
    #image_list = ['IMG_105.jpg','IMG_153.jpg','IMG_23.jpg','IMG_283.jpg','IMG_306.jpg','IMG_78.jpg','IMG_93.jpg']
    image_list = []
    for file in os.listdir(image_path):

        image_list.append(os.path.join(image_path, file))

    data_analysis(use_gpu, model_path, image_list,resize=1024)


