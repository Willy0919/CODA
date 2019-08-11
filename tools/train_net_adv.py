#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Willy, Weiyuan

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.lib.opt.train_adv import TrainModel
from src.lib.network.cn import vgg16
import yaml
from easydict import EasyDict as edict
import sys,getopt

def initGenFeatFromCfg(cfg_file):
    # Load cfg parameter from yaml file
    with open(cfg_file, 'r') as f:
        cfg = edict(yaml.load(f))

    # Fist load the dataset name
    dataset = cfg.DATASET

    # Set values
    #path
    data_path = cfg[dataset].DATA_PATH
    target_data_path = cfg[dataset].TARGET_DATA_PATH
    tensor_server_path = cfg[dataset].TENSOR_BOARD_PATH
    pre_trained_path = cfg[dataset].PRE_TRAINED_PATH

    #network
    batch_size = cfg[dataset].BATCH_SIZE
    lr = float(cfg[dataset].LEARNING_RATE)
    epoch_num = cfg[dataset].EPOCH_NUM
    steps = cfg[dataset].STEPS
    decay_rate = cfg[dataset].DECAY_RATE
    start_epoch = cfg[dataset].START_EPOCH
    snap_shot = cfg[dataset].SNAP_SHOT
    resize = cfg[dataset].RESIZE
    test_size = cfg[dataset].TEST_SIZE

    return dataset, data_path,target_data_path, tensor_server_path, pre_trained_path, batch_size, lr, epoch_num, steps, decay_rate, start_epoch, snap_shot, resize, test_size


def dispHelp():
    print("======================================================")
    print("                       Usage")
    print("======================================================")
    print("\t-h display this message")
    print("\t--cfg <config file yaml>")

def main(argv):
    cfg_file = 'model/shanghaitech_adv.yml'

    # Get parameters
    try:
        opts, _ = getopt.getopt(argv, "h:", ["cfg="])
    except getopt.GetoptError:
        dispHelp()
        return

    for opt, arg in opts:
        if opt == '-h':
            dispHelp(argv[0])
            return
        elif opt in ("--cfg"):
            cfg_file = arg

    print("Loading configuration file: ", cfg_file)
    (dataset, data_path,target_data_path, tensor_server_path, pre_trained_path, batch_size, lr, epoch_num, steps, decay_rate, start_epoch, snap_shot, resize, test_size) = initGenFeatFromCfg(cfg_file)

    print("Choosen parameters:")
    print("-------------------")
    print("Dataset: ", dataset)
    print("Source Data location: ", data_path)
    print("Target Data location: ", target_data_path)
    print("Tensorboard server root: ", tensor_server_path)
    print("Pre-trained density model path:", pre_trained_path['density'])
    print("Pre-trained discriminator model path:", pre_trained_path['discriminator'])
    print("Batch size:", batch_size)
    print("Learning rate: ", lr)
    print("Total epoch number: ", epoch_num)
    print("Learning rate steps: ", steps)
    print("Learning rate decay rate: ", decay_rate)
    print("Start epoch: ", start_epoch)
    print("Snap shot: ", snap_shot)
    print("Image resize: ", resize)
    print("Train size: {} ;Validation size: {}".format(test_size['train'],test_size['test']) )
    print("")
    print("===================")
    print("")

    tm = TrainModel(data_path=data_path,target_data_path=target_data_path, batchsize=batch_size, lr=lr, epoch=epoch_num, snap_shot=snap_shot,
                        server_root_path=tensor_server_path, start_epoch=start_epoch, steps=steps,
                        decay_rate=decay_rate, branch=vgg16, pre_trained=pre_trained_path,resize=resize, test_size=test_size)

    tm.run()



if __name__ == '__main__':
    main(sys.argv[1:])
