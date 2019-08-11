#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Willy, Weiyuan

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from src.lib.opt.lr_policy import update_lr
from src.lib.dataset.shanghaitech import HeadCountDataset,IsColor,RandomFlip,PreferredSize,ToTensor,Normalize,NineCrop
from torchvision import transforms
from torch.utils.data import DataLoader
from src.lib.network.cn import vgg16
import src.lib.utils.Logger as logger
import os, time, sys
from skimage import transform as sk_transform

class TrainModel(object):
    def __init__(self,data_path,batchsize,lr,epoch, snap_shot, server_root_path,start_epoch = 0,steps = [],decay_rate = 0.1, branch = vgg16, pre_trained=None,resize=896, val_size=50):
        self.use_gpu = torch.cuda.is_available()
        self.batchsize = batchsize
        self.val_size = val_size
        #tensorboard log and step log
        self.logger = logger.Logger(os.path.join(server_root_path, 'log', branch.name))
        self.log_path = os.path.join(server_root_path, 'log', branch.name + '.log')
        f = open(self.log_path, 'a')
        #f.truncate()
        f.close()

        self.save_path = os.path.join(server_root_path, 'ex', branch.name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.train_dataset = HeadCountDataset(data_file=os.path.join(data_path,'train_data.txt'),use_pers=False,use_attention=False,
                                         transform=transforms.Compose([IsColor(True), NineCrop(), RandomFlip(),PreferredSize(resize), ToTensor(), Normalize()]))


        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batchsize, shuffle=True, num_workers=32)

        self.val_dataset = HeadCountDataset(data_file=os.path.join(data_path,'test_data.txt'),use_pers=False,use_attention=False,
                                            transform=transforms.Compose(
                                                [IsColor(True),PreferredSize(1024), ToTensor(use_att=False), Normalize()]))
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=4, shuffle=False, num_workers=32)
        self.dataloader = {'train':self.train_dataloader,'val':self.val_dataloader}
        self.model = branch()
        self.model.create_architecture()
        self.start_epoch = start_epoch

        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=0.9)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loadModel(pre_trained)
        self.criterion_dens = nn.MSELoss(size_average=False)
        self.criterion_count = nn.L1Loss(size_average=False)
        self.num_epoch = epoch
        self.snap_shot = snap_shot
        self.steps = steps
        self.decay_rate = decay_rate

    def loadModel(self, model_path=None):
        if model_path != '':
            pretrained_dict = torch.load(model_path)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            for k in pretrained_dict:
                print('key:',k)
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            print('Load model:{}'.format(model_path))


    def saveModel(self, save_path,epoch):
        model_path = save_path + '/{}.model'.format(str(epoch))
        torch.save(self.model.state_dict(), model_path)

    def train_model(self,model, optimizer, num_epochs=25, val_size=50):

        best_mae = sys.maxsize
        for epoch in range(self.start_epoch,num_epochs+1):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            for phase in ['train','val']:
                if phase == 'train':
                    optimizer = update_lr(optimizer, epoch, self.steps, self.decay_rate)
                    model.train(True)
                else:
                    if epoch % val_size != 0:
                        continue
                    model.train(False)
                    model.eval()

                running_loss = 0.0
                running_mse = 0.0
                running_mae = 0.0
                totalnum = 0
                for idx,data in enumerate(self.dataloader[phase]):
                    image = data['image']
                    densityMap = data['densityMap']

                    if self.use_gpu:
                        image, densityMap = Variable(image.cuda(),requires_grad=False), Variable(densityMap.cuda(),requires_grad=False)

                        self.model = self.model.cuda()
                    else:
                        image, densityMap = Variable(image), Variable(densityMap)

                    #print('image size:', image.size(), densityMap.size())
                    optimizer.zero_grad()

                    duration = time.time()

                    predDensityMap = model(image)
                    #Resize

                    predDensityMap = torch.squeeze(predDensityMap)
                    loss = self.criterion_dens(predDensityMap, densityMap)
                    time_elapsed = time.time() - duration

                    outputs_np = predDensityMap.data.cpu().numpy()
                    densityMap_np = (densityMap).data.cpu().numpy()


                    pre_dens = np.sum(outputs_np.reshape((outputs_np.shape[0],-1)),1)
                    gt_count = np.sum(densityMap_np.reshape((densityMap_np.shape[0],-1)),1)

                    totalnum += outputs_np.shape[0]

                    running_mae += np.sum(np.abs(pre_dens-gt_count))
                    running_mse += np.sum((pre_dens-gt_count)**2)


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        # if idx == 20 and epoch%100==0:
                        #     showTest(image.data.cpu(),outputs.data.cpu(),densityMap.data.cpu())
                        if idx%20 == 0 and phase=='val':
                            f = open(self.log_path, 'a')
                            f.write('-------------------density count----------------------\n')
                            f.write("epoch: %4d, step: %4d, Time: %.4fs, gt cnt: %4.1f, density cnt: %4.1f, loss: %4.4f\n" %
                                  (epoch, idx, time_elapsed, np.sum(gt_count), np.sum(pre_dens),  loss.data[0]))
                            f.write('---------------------------------------------------\n')

                            f.close()


                    running_loss += loss.data[0]

                epoch_loss = running_loss / totalnum
                epoch_mae = running_mae/totalnum
                epoch_mse = np.sqrt(running_mse/totalnum)
                if epoch % self.snap_shot == 0:
                    self.saveModel(self.save_path,epoch)
                    if best_mae > epoch_mae and phase=='val':
                        best_mae = epoch_mae
                        f = open(self.log_path, 'a')
                        f.write("+++++++++++++++++++++best density epoch: {} ++++++++++++++++++++++++\n".format(epoch))
                        f.write('{} Loss: {:.4f} MAE: {:.4f} MSE: {:.4f}\n'.format(phase, epoch_loss, epoch_mae, epoch_mse))
                        f.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                        f.close()


                print('{} Loss: {:.4f} density MAE: {:.4f}  density MSE: {:.4f}'.format(phase, epoch_loss,epoch_mae,epoch_mse))
                self.logger.scalar_summary('Temporal/'+ phase +'/loss', epoch_loss, epoch)
                self.logger.scalar_summary('Temporal/' + phase +'/dens_MAE', epoch_mae, epoch)
                self.logger.scalar_summary('Temporal/' + phase + '/dens_MSE', epoch_mse, epoch)

    def run(self):
        self.train_model(self.model,self.optimizer,self.num_epoch, self.val_size)


if __name__ == '__main__':
    tm = TrainModel(path='data/ShanghaiTech/part_A_final/',batchsize=32,lr=1e-5,epoch=1000,\
                    snap_shot=5,server_root_path='exp/ShanghaiTech',start_epoch=600,steps=[],\
                    decay_rate=0.1, branch=vgg16, pre_trained='exp/ShanghaiTech/ex/610.model')
    tm.run()
