from re import I
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, glob
import json
from ossid.datasets.render_dataset import loadHdf5
from tqdm import tqdm

from multiprocessing import Pool

def main():
    # input_root = "/home/qiaog/datasets/render/shapenetcc2/"
    input_root = "/home/qiaog/datasets/render/shapenetccbop"
    output_file = os.path.join(input_root, "object2files.json")
    all_paths = glob.glob(os.path.join(input_root, "*.hdf5"))
    all_paths.sort()

    object2files = {}

    for path in tqdm(all_paths):
        data = loadHdf5(path)
        
        segcolormap = data['segcolormap']
        segmap = data['segmap']
        for inst in segcolormap:
            cate_id = int(inst['category_id'])

            # Ignore the case of multiple instance of one object for now
            if len([_ for _ in segcolormap if _['category_id'] == inst['category_id']]) > 1:
                continue

            inst_id = int(inst['idx'])
            cate_mask_channel = int(inst['channel_class'])
            inst_mask_channel = int(inst['channel_instance'])

            # check if it is a background
            if cate_id == 0:
                continue

            inst_mask = np.logical_and(segmap[:, :, cate_mask_channel] == cate_id, segmap[:, :, inst_mask_channel] == inst_id)
            
            n_vispx = inst_mask.sum()
            
            if n_vispx < 1000:
                continue
                
            if cate_id not in object2files:
                object2files[cate_id] = []
                
            object2files[cate_id].append(path.split("/")[-1].split(".")[0])
            
    json.dump(object2files, open(output_file, "w"), indent=2)


if __name__ == "__main__":
    main()