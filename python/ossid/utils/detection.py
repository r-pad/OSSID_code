import os, argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

from zephyr.utils.bop_dataset import BopDataset

from ossid.utils.results import loadResult

from pathlib import Path

def saveDetResults(results, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for k, v in results.items():
        scene_id, im_id = k
        filename = "s%06d_i%06d.txt" % (scene_id, im_id)
        path = os.path.join(folder, filename)
        
        with open(path, 'w') as f:
            for box in v:
                if len(box) == 5:
                    obj_id, x1, y1, x2, y2 = box
                    f.write("obj_%06d %d %d %d %d\n" % (obj_id, x1, y1, x2, y2))
                elif len(box) == 6:
                    obj_id, x1, y1, x2, y2, s = box
                    f.write("obj_%06d %04f %d %d %d %d\n" % (obj_id, s, x1, y1, x2, y2))

    return

def saveGTResults(bop_dataset, dataset_name, save_root=Path("./DetResults")):
    results = {}

    for idx in trange(len(bop_dataset)):
        bop_data = bop_dataset[idx]

        obj_id = bop_data['obj_id']
        scene_id = bop_data['scene_id']
        im_id = bop_data['im_id']

        mask = bop_data['mask_gt_visib']

        # Will be in (x1, y1, x2, y2) format, where x is rightward and y is downward
        # This corresponding to <left> <top> <right> <bottom>
        h, w = mask.shape
        mask_pixels = np.stack(mask.nonzero(), axis=1)
        y1, x1 = mask_pixels.min(0)
        y2, x2 = mask_pixels.max(0)

        key = (scene_id, im_id)
        if key not in results:
            results[key] = []

        results[key].append((obj_id, x1, y1, x2, y2))
        
    saveDetResults(results, save_root / ("gt-%s/" % dataset_name))

def saveLmoYcbvGT(save_root=Path("./DetResults"), bop_root="/home/qiaog/datasets/bop/"):
    if not os.path.exists(save_root / ("gt-%s/" % "lmo")):
        print("Saving lmo detetcion GT. This may take some time...")
        test_args = argparse.Namespace()
        test_args.bop_root = bop_root
        test_args.dataset_name = "lmo"
        test_args.split_name = "bop_test"
        test_args.split = "test"
        test_args.split_type = None
        test_args.model_type = None
        test_args.skip = 1

        bop_dataset = BopDataset(test_args)

        saveGTResults(bop_dataset, 'lmo', save_root)
    else:
        print("GT Det for lmo already exists")
    
    if not os.path.exists(save_root / ("gt-%s/" % "ycbv")):
        print("Saving ycbv detetcion GT. This may take some time...")
        test_args = argparse.Namespace()
        test_args.bop_root = bop_root
        test_args.dataset_name = "ycbv"
        test_args.split_name = "bop_test"
        test_args.split = "test"
        test_args.split_type = None
        test_args.model_type = None
        test_args.skip = 1

        bop_dataset = BopDataset(test_args)

        saveGTResults(bop_dataset, 'ycbv', save_root)
    else:
        print("GT Det for ycbv already exists")

def runMapEval(gt_folder, det_folder):
    map_script_root = Path("/home/qiaog/src/mAP/")
    
    gt_dst = map_script_root / 'input' / "ground-truth"
    det_dst = map_script_root / 'input' / "detection-results"

    if os.path.islink(gt_dst):
        os.remove(gt_dst)
    if os.path.islink(det_dst):
        os.remove(det_dst)

    os.symlink(os.path.abspath(gt_folder), gt_dst)
    os.symlink(os.path.abspath(det_folder), det_dst)

    process = subprocess.run(['python', 'main.py', '--no-animation', '--no-plot'],
                                capture_output=True,
                                cwd=map_script_root,
                                )

    output = process.stdout.decode("utf-8") 
    
    results = {}
    for line in output.split("\n"):
        line = line.rstrip()
        parts = line.split(" ")
        if len(parts) == 4:
            obj = parts[2]
            AP = float(parts[0][:-1])
            results[obj] = AP
        elif len(parts) == 3:
            results['mAP'] = float(parts[2][:-1])

    return results
        
def evalFinetuneResults(result_or_path, dataset_name, tmp_root=Path("./DetResults") ):
    if type(result_or_path) is str or type(result_or_path) is Path:
        result = loadResult(result_or_path, None)
    else:
        result = result_or_path

    if type(result) is pd.DataFrame:
        iterator = result.iterrows()
    elif type(result) is list:
        iterator = result
    else:
        raise Exception("Unknown result type:", type(result))
    
    det_results = {}

    for r in iterator:
        try:
            r = r[1]
        except KeyError:
            pass
        obj_id, scene_id, im_id = r['obj_id'], r['scene_id'], r['im_id']
        dtoid_bbox, dtoid_score = r['dtoid_bbox'], r['dtoid_score']

        key = (scene_id, im_id)
        if key not in det_results:
            det_results[key] = []

#         for i in range(len(dtoid_bbox)):
        for i in range(1):
            x1, y1, x2, y2 = dtoid_bbox[i]
            s = dtoid_score[i]
            det_results[key].append((obj_id, x1, y1, x2, y2, s))

    save_results_path = tmp_root / ("tmp-%s" % dataset_name)
    saveDetResults(det_results, save_results_path)
    if dataset_name == 'lmo':
        gt_path = tmp_root / "gt-lmo"
    elif dataset_name == 'ycbv':
        gt_path = tmp_root / "gt-ycbv"
    else:
        raise Exception("Unknown dataset name:", dataset_name)
        
    det_results = runMapEval(gt_path, save_results_path)
    
    print("Detection mAP metrics:")
    print("Per-object detection AP", det_results)
    print("Detection mAP:", det_results['mAP'])
    
    print()

    return det_results['mAP']