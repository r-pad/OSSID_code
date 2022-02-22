# Model Checkpoints 

This file describes the files that should be contained in this folder and their usage. 

* `dtoid_conf_lmo.yaml`: configuration file for training and evaluation on LM-O dataset. 
* `dtoid_conf_ycbv.yaml`: configuration file for training and evaluation on YCB-V dataset. 
* `dtoid_pretrained.ckpt`: weights of DTOID detector, trained on our own synthectic dataset that does not include LM-O and YCB-V objects. This model is used for OSSID evaluation on YCB-V. 
* `dtoid_transductive_lmo.ckpt`: weights of DTOID detector, trained on our own synthectic dataset and then self-supervised finetuned on LM-O BOP test set with labels provided by Zephyr.
* `dtoid_transductive_ycbv.ckpt`: weights of DTOID detector, trained on our own synthectic dataset and then self-supervised finetuned on YCB-V BOP test set with labels provided by Zephyr.
* `dtoid_pretrained_original.pth.tar`: weights of DTOID detector, provided by the [Mercier et al](https://github.com/jpmerc/DTOID). This model sees YCB-V objects during training. This model is used for OSSID evaluation on LM-O only. 
* `final_lmo.ckpt`: weights for Zephyr model on LM-O objects. 
* `final_ycbv.ckpt`: weights for Zephyr model on YCB-V objects, trained on objects with odd ID, for evaluation on objects with even ID. 
* `final_ycbv_valodd.ckpt`: weights for Zephyr model on YCB-V objects, trained on objects with even ID, for evaluation on objects with odd ID. 