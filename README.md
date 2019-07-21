# CODA
Implementation for [CODA: Counting Objects via Scale-aware Adversarial Adaption](https://arxiv.org/pdf/1903.10442.pdf) 


```
* data
       * dataset folders
       
* exp
       * experiments logs
       
* model
       * %dataset name%.yml: dataset configurations
       
* src/lib
       * dataset
            * DataPrepare.py: preparing the train/val/test list for training/testing
            * mall/shanghaitech/trancos.py: dataset redefinition
       * network
            * network.py: network definition
       * opt
            * train.py: training operation
            * lr_policy.py: lr policy 
       * utils
            * image_opt.py: image visualization
            * Logger.py: tensorboard logger
        
* tools
       * demo.py
       * train_net.py
```
