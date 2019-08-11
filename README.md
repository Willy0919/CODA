# CODA
Pytorch implementation for [CODA: Counting Objects via Scale-aware Adversarial Adaption](https://arxiv.org/pdf/1903.10442.pdf) 


```
* data
       * dataset folders
       
* exp
       * experiments logs
       
* model
       * %dataset name%.yml: dataset configurations
       * %dataset name%_adv.yml: dataset configurations for adversarial training and testing
       
* src/lib
       * dataset
            * DataPrepare.py: preparing the train/val/test list for training/testing
            * mall/shanghaitech/trancos.py: dataset redefinition
       * network
            * cn.py: counting network definition
            * discriminator.py: discriminator definition
       * opt
            * train.py: training operation for counting network
            * train_adv.py: training operation for adversarial training
            * lr_policy.py: lr policy 
       * utils
            * image_opt.py: image visualization
            * Logger.py: tensorboard logger
        
* tools
       * demo.py
       * demo_adv.py
       * train_net.py
       * train_net_adv.py
```


## pretrain CN

```
python tools/train_net.py
```

## density adaption

```
python tools/train_net_adv.py
```
