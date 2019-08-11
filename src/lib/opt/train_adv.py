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
from src.lib.dataset.shanghaitech import HeadCountDataset,IsColor,RandomFlip,PreferredSize,ToTensor,Normalize,NineCrop,Multiscale
from torchvision import transforms
from torch.utils.data import DataLoader
from src.lib.network.cn import vgg16
from src.lib.network.discriminator import Discriminator
import src.lib.utils.Logger as logger
import os, time, sys
from skimage import transform as sk_transform

class TrainModel(object):
    def __init__(self,data_path,target_data_path,batchsize,lr,epoch, snap_shot, server_root_path,start_epoch = 0,steps = [],decay_rate = 0.1, branch = vgg16, pre_trained=None,resize=896, test_size={'train':10, 'test':100}):
        self.use_gpu = torch.cuda.is_available()
        self.batchsize = batchsize
        self.test_size = test_size
        self.use_multiscale = True
        #tensorboard log and step log
        self.logger = logger.Logger(os.path.join(server_root_path, 'log', branch.name))
        self.log_path = os.path.join(server_root_path, 'log', branch.name + '.log')
        f = open(self.log_path, 'a')
        #f.truncate()
        f.close()
        self.lr_D = 1e-3
        self.save_path = os.path.join(server_root_path, 'ex', branch.name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        train_dataset = HeadCountDataset(epoch,'train_adv',os.path.join(data_path,'train_data.txt'),use_pers=False,use_attention=False,
                                         transform=transforms.Compose([IsColor(True), NineCrop(), RandomFlip(),PreferredSize(resize), ToTensor(use_att=False), Normalize()]))
        target_dataset = HeadCountDataset(epoch,'train_adv',os.path.join(target_data_path, 'train_data.txt'), use_pers=False,
                                              use_attention=False,
                                              transform=transforms.Compose(
                                                  [IsColor(True), NineCrop(), RandomFlip(),Multiscale(cropscale=[0.5,0.75]), PreferredSize(resize,use_multiscale=self.use_multiscale),
                                                   ToTensor(use_att=False,use_multiscale=self.use_multiscale), Normalize(use_multiscale=self.use_multiscale)]))

        train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=32)
        target_dataloader = DataLoader(target_dataset, batch_size=batchsize, shuffle=True, num_workers=32)

        val_dataset = HeadCountDataset(epoch,'test',os.path.join(target_data_path,'test_data.txt'),use_pers=False,use_attention=False,
                                            transform=transforms.Compose(
                                                [IsColor(True),PreferredSize(1024), ToTensor(use_att=False), Normalize()]))
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=32)
        self.dataloader = {'train':train_dataloader,'target':target_dataloader,'val':val_dataloader}
        print("branch:",branch)
        self.model = branch()
        self.model.create_architecture()
        self.start_epoch = start_epoch
        self.model_D = Discriminator(1)

        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=0.9)

        self.optimizer_D = optim.Adam(self.model_D.parameters(), lr=self.lr_D, betas=(0.9, 0.99))
        #self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        #load model
        if pre_trained['density'] != '':
            self.loadModel(self.model, pre_trained['density'])
        if pre_trained['discriminator'] != '':
            self.loadModel(self.model_D,pre_trained['discriminator'])

        self.criterion_dens = nn.MSELoss(size_average=False)
        #self.criterion_count = nn.L1Loss(size_average=False)
        self.criterion_disc = nn.BCEWithLogitsLoss()
        self.criterion_rank = nn.MarginRankingLoss()
        self.num_iter = epoch #250000
        self.snap_shot = snap_shot
        self.steps = steps
        self.decay_rate = decay_rate
        self.power = 0.9
        self.source_label = 0
        self.target_label = 1
        self.lambda_adv = 0.001

    def lr_poly(self,base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def adjust_learning_rate_D(self,optimizer, i_iter):
        lr = self.lr_poly(self.lr_D, i_iter, self.num_iter, self.power)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def loadModel(self,model, model_path=None):
        if model_path != '':
            pretrained_dict = torch.load(model_path)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            for k in pretrained_dict:
                print('key:',k)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print('Load model:{}'.format(model_path))


    def saveModel(self,model, save_path,epoch):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        model_path = save_path + '/{}.model'.format(str(epoch))
        torch.save(model.state_dict(), model_path)

    def train_model(self):

        #preparing
        trainloader_iter = enumerate(self.dataloader['train'])
        targetloader_iter = enumerate(self.dataloader['target'])
        #testloader_iter = enumerate(self.dataloader['val'])

        best_mae = sys.maxsize
        loss_dens_value = 0.0
        loss_adv_value = 0.0
        loss_D_value = 0.0

        running_mse = 0.0
        running_mae = 0.0
        totalnum = 0
        iter_count = 0
        for epoch in range(self.start_epoch,self.num_iter+1):
            iter_count = iter_count + 1
            print('Iteration {}/{}'.format(epoch + 1, self.num_iter))
            # training


            self.optimizer = update_lr(self.optimizer, epoch, self.steps, self.decay_rate)
            self.model.train(True)
            self.model_D.train(True)
            self.adjust_learning_rate_D(self.optimizer_D, epoch)
            self.optimizer.zero_grad()
            self.optimizer_D.zero_grad()

            # train G
            # dont't accumulate grads in D

            for param in self.model_D.parameters():
                param.requires_grad = False

            # train with source

            _, data = next(trainloader_iter)
            image = data['image']
            Dmap = data['densityMap']
            if self.use_gpu:
                image, Dmap = Variable(image.cuda(), requires_grad=False), Variable(Dmap.cuda(),
                                                                                          requires_grad=False)
                self.model = self.model.cuda()
                self.model_D = self.model_D.cuda()
            else:
                image, Dmap = Variable(image), Variable(Dmap)

            duration = time.time()
            predDmap = self.model(image)
            loss = self.criterion_dens(torch.squeeze(predDmap,1), Dmap)
            time_elapsed = time.time() - duration

            outputs_np = predDmap.data.cpu().numpy()
            Dmap_np = (Dmap).data.cpu().numpy()

            # calculate iter size
            iter_size = outputs_np.shape[0]
            totalnum += iter_size
            #print('iter size:{} totalnum:{}'.format(iter_size, totalnum))

            # backpropogate G
            loss.backward()
            #self.optimizer.step()
            loss_dens_value += loss.data[0]/iter_size

            # train with target
            _, data_t = next(targetloader_iter)
            image_t = data_t['image']
            if self.use_multiscale:
                multi_img = data_t['scale_images']

            #Dmap = data['densityMap']
            if self.use_gpu:
                image_t = Variable(image_t.cuda(), requires_grad=False)
                if self.use_multiscale:
                    multi_img = Variable(multi_img.cuda(), requires_grad=False)
                self.model = self.model.cuda()
                self.model_D = self.model_D.cuda()
            else:
                image_t = Variable(image_t)
                if self.use_multiscale:
                    multi_img = Variable(multi_img)

            predDmap_t = self.model(image_t)
            D_out_t = self.model_D(predDmap_t)
            loss = self.lambda_adv * self.criterion_disc(D_out_t, Variable(
                torch.FloatTensor(D_out_t.data.size()).fill_(self.source_label)).cuda())

            if self.use_multiscale:
                multi_imgs = torch.chunk(multi_img,multi_img.size()[1],dim=1)
                #add sub_img for constrains
                predDmap_t_subs = []
                D_out_t_subs = []
                for sub_img in multi_imgs:
                    sub_img = torch.squeeze(sub_img)
                    #print(multi_img.size(), image_t.size(),sub_img.size())
                    predDmap_t_sub = self.model(sub_img)
                    predDmap_t_subs.append(predDmap_t_sub)
                    D_out_t_sub = self.model_D(predDmap_t_sub)
                    D_out_t_subs.append(D_out_t_sub)


                # add adversarial loss(density loss & rankloss)
                for i in range(len(multi_imgs)):
                    loss += self.lambda_adv*self.criterion_disc(D_out_t_subs[i],Variable(torch.FloatTensor(D_out_t_subs[i].data.size()).fill_(self.source_label)).cuda())
                    if i == 0:
                        pred_cnt = torch.sum(predDmap_t.reshape(predDmap_t.size(0),-1),dim=1)

                        pred_cnt_sub = torch.sum(predDmap_t_subs[i].reshape(predDmap_t_subs[i].size(0), -1), dim=1)
                        loss += self.lambda_adv*self.criterion_rank(pred_cnt_sub,pred_cnt,Variable(torch.Tensor(pred_cnt.data.size()).fill_(-1)).cuda())
                    else:
                        pred_cnt_sub1 = torch.sum(predDmap_t_subs[i-1].reshape(predDmap_t_subs[i-1].size(0), -1), dim=1)
                        pred_cnt_sub2 = torch.sum(predDmap_t_subs[i].reshape(predDmap_t_subs[i].size(0), -1), dim=1)
                        loss += self.lambda_adv * self.criterion_rank(pred_cnt_sub2, pred_cnt_sub1, Variable(
                            torch.Tensor(pred_cnt_sub1.data.size()).fill_(-1)).cuda())


            loss.backward()
            loss_adv_value += loss / iter_size
            #self.optimizer.step()

            #calculating mae & mse
            pre_dens = np.sum(outputs_np.reshape((outputs_np.shape[0], -1)), 1)
            gt_count = np.sum(Dmap_np.reshape((Dmap_np.shape[0], -1)), 1)

            running_mae += np.sum(np.abs(pre_dens - gt_count))
            running_mse += np.sum((pre_dens - gt_count) ** 2)

            # train D

            # bring back requires_grad
            for param in self.model_D.parameters():
                param.requires_grad = True
            # train with source
            predDmap = predDmap.detach()

            D_out = self.model_D(predDmap)

            loss = self.criterion_disc(D_out,Variable(torch.FloatTensor(D_out.data.size()).fill_(self.source_label)).cuda())

            loss.backward()
            loss_D_value += loss.data[0]/iter_size

            # train with target

            predDmap_t = predDmap_t.detach()
            D_out_t = self.model_D(predDmap_t)
            loss = self.criterion_disc(D_out_t,
                                       Variable(torch.FloatTensor(D_out_t.data.size()).fill_(self.target_label)).cuda())
            loss.backward()

            loss_D_value += loss.data[0] / iter_size
            # add sub img loss(density loss)
            if self.use_multiscale:
                for i in range(len(predDmap_t_subs)):
                    predDmap_t_subs[i] = predDmap_t_subs[i].detach()
                    D_out_t_sub=self.model_D(predDmap_t_subs[i])
                    loss = self.criterion_disc(D_out_t_sub,Variable(torch.FloatTensor(D_out_t_sub.data.size()).fill_(self.target_label)).cuda())
                    loss.backward()
                    loss_D_value += loss.data[0] /iter_size

            # optimizer
            self.optimizer.step()
            self.optimizer_D.step()

            print('Train density Loss: {:.4f} Density Adversarial Loss: {:.4f}  Discriminator Loss: {:.4f}'.format(loss_dens_value/iter_count, loss_adv_value/iter_count, loss_D_value/iter_count))

            #test mae & mse on train set
            if epoch %self.test_size['train'] == 0:
                epoch_mae = running_mae / totalnum
                epoch_mse = np.sqrt(running_mse /totalnum)
                print('==============Training Iteration:{} MAE: {:.4f} MSE: {:.4f}'.format(epoch,epoch_mae, epoch_mse))
                print('==Train density Loss: {:.4f} Density Adversarial Loss: {:.4f}  Discriminator Loss: {:.4f}'.format(
                    loss_dens_value / iter_count, loss_adv_value / iter_count, loss_D_value / iter_count))

                f = open(self.log_path, 'a')
                f.write('Iteration:{} MAE: {:.4f} MSE: {:.4f}\n'.format(epoch, epoch_mae, epoch_mse))
                f.close()
                running_mae = 0.0
                running_mse = 0.0
                totalnum= 0.0
                loss_dens_value = 0.0
                loss_adv_value = 0.0
                loss_D_value = 0.0
                iter_count = 0

            self.logger.scalar_summary('Temporal/train_density_loss', loss_dens_value, epoch)
            self.logger.scalar_summary('Temporal/train_adv_loss', loss_adv_value, epoch)
            self.logger.scalar_summary('Temporal/train_D_loss', loss_D_value, epoch)


            #testing
            if epoch % self.test_size['test'] == 0:
                self.model.train(False)
                self.model.eval()
                self.model_D.train(False)
                self.model_D.eval()

                totalnum_test = 0.0
                running_mse_test = 0.0
                running_mae_test = 0.0
                running_loss_test = 0.0

                for idx,data in enumerate(self.dataloader['val']):

                    image = data['image']
                    densityMap = data['densityMap']

                    if self.use_gpu:
                        image, densityMap = Variable(image.cuda(), requires_grad=False), Variable(densityMap.cuda(),
                                                                                                  requires_grad=False)

                        self.model = self.model.cuda()
                    else:
                        image, densityMap = Variable(image), Variable(densityMap)

                    duration = time.time()

                    predDensityMap = self.model(image)


                    loss = self.criterion_dens(torch.squeeze(predDensityMap,1), densityMap)
                    time_elapsed = time.time() - duration

                    outputs_np = predDensityMap.data.cpu().numpy()
                    densityMap_np = (densityMap).data.cpu().numpy()


                    pre_dens = np.sum(outputs_np.reshape((outputs_np.shape[0],-1)),1)
                    gt_count = np.sum(densityMap_np.reshape((densityMap_np.shape[0],-1)),1)

                    totalnum_test += outputs_np.shape[0]

                    running_mae_test += np.sum(np.abs(pre_dens-gt_count))
                    running_mse_test += np.sum((pre_dens-gt_count)**2)

                    running_loss_test += loss.data[0]

                epoch_loss = running_loss_test / totalnum_test
                epoch_mae = running_mae_test/totalnum_test
                epoch_mse = np.sqrt(running_mse_test/totalnum_test)

                f = open(self.log_path, 'a')
                print('*****************Test MAE: {:.4f} MSE: {:.4f}'.format(epoch_mae, epoch_mse))

                f.write('****Test Iterator:{} MAE: {:.4f} MSE: {:.4f}\n'.format(epoch, epoch_mae, epoch_mse))
                f.close()
                if epoch % self.snap_shot == 0:
                    self.saveModel(self.model,os.path.join(self.save_path, 'Generator'),epoch)
                    self.saveModel(self.model_D,os.path.join(self.save_path, 'Discriminator'),epoch)
                    if best_mae > epoch_mae:
                        best_mae = epoch_mae
                        f = open(self.log_path, 'a')
                        f.write("+++++++++++++++++++++best density epoch: {} ++++++++++++++++++++++++\n".format(epoch))
                        f.write('Loss: {:.4f} MAE: {:.4f} MSE: {:.4f}\n'.format(epoch_loss, epoch_mae, epoch_mse))
                        f.write("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                        f.close()

                self.logger.scalar_summary('Temporal/test_density_loss', epoch_loss, epoch)
                self.logger.scalar_summary('Temporal/test_mae', epoch_mae, epoch)
                self.logger.scalar_summary('Temporal/test_mse', epoch_mse, epoch)


    def run(self):
        self.train_model()


if __name__ == '__main__':
    tm = TrainModel(path='data/ShanghaiTech/part_A_final/',batchsize=32,lr=1e-5,epoch=1000,\
                    snap_shot=5,server_root_path='exp/ShanghaiTech',start_epoch=600,steps=[],\
                    decay_rate=0.1, branch=vgg16, pre_trained='exp/ShanghaiTech/ex/610.model')
    tm.run()
