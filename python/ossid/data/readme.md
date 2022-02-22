# Data used for OSSID

This file describes the files that should be contained in this folder

* `template_LMO_DTOID` (unzipped from `template_LMO_DTOID.zip`): template images for LM-O objects, provided by [Mercier et al.](https://github.com/jpmerc/DTOID). Note that the object models used to generate this template set is different than provided by BOP Challenge. We found that this template set has better performance. 
* `template_YCBV_BOP` (unzipped from `template_YCBV_BOP.zip`): template images for YCB-V objects, rendered using the YCB-V object mesh model provided in [BOP Challenge](https://bop.felk.cvut.cz/datasets/). 
* `zephyr_model_data` (unzipped from `zephyr_model_data.zip`): model point clouds for YCB-V and LM-O. Each file contains the xyz coordinates, color and normal vectors for point clouds. 
* `lmo_boptest_zephyr_result.pkl`: a pickle file containing Zephyr pose estimation results on LM-O BOP test set. 
* `test_ycbv_boptest_zephyr_result_unseen.pkl`: a pickle file containing Zephyr pose estimation results on YCB-V BOP test set. 
* `ycbv_grid.zip`: a zip file containing SIFT featurization of YCB-V objects. This file needs to be unzipped at ycbv dataset folder (`BOP_DATASETS_ROOT/ycbv/`). 