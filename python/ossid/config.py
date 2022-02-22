import os

# The path to OSSID source code
OSSID_ROOT = "/home/qiaog/src/OSSID/OSSID_cleanup" 
# The path to BOP dataset root folder, which contains folder named `lmo` and `ycbv`
BOP_DATASETS_ROOT = "/home/qiaog/datasets/bop/"
# Path to bop toolkit source code
BOP_TOOLKIT_PATH = "/home/qiaog/src/bop_toolkit"
# Path to a folder where all evaluation results (in BOP format and generated by BOP evaluation script) will be stored
BOP_RESULTS_FOLDER = "/home/qiaog/src/OSSID/bop_results"


'''The following lines should not be changed'''
OSSID_ROOT = os.path.join(OSSID_ROOT, "python", "ossid")
OSSID_CKPT_ROOT = os.path.join(OSSID_ROOT, "ckpts")
OSSID_DATA_ROOT = os.path.join(OSSID_ROOT, "data")
OSSID_RESULT_ROOT = os.path.join(OSSID_ROOT, "scripts/finetune_results")
OSSID_DET_ROOT = os.path.join(OSSID_ROOT, "scripts/det_tmp")