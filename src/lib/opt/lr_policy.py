#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Willy

def update_lr(optimizer, epoch, steps, decay_rate):
    if epoch in steps:
        for param_group in optimizer.param_groups:
            print('before_lr',param_group['lr'])
            param_group['lr'] *= decay_rate
            print('after_lr',param_group['lr'])
    return optimizer
