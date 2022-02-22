from datetime import time
import os
import csv
import numpy as np
import pandas as pd
import subprocess

from ossid.config import BOP_TOOLKIT_PATH

def saveResultsBop(
        results, output_folder, result_name, dataset_name, 
        split_name='test', pose_key = 'pose', score_key='score', time_key = 'time',
        input_unit = 'm', # If input is meter, it will be convert to milimeter
        run_eval_script=False, # if True, run the BOP evaluation script
    ):
    '''
    Convert a list of dict containing the pose estimation results to a csv for BOP evaluation script.
    '''
    result_name = result_name.replace("_", "-")
    output_filename = "%s_%s-%s.csv" % (result_name, dataset_name, split_name)
    output_path = os.path.join(
        output_folder,
        output_filename
    )

    csv_file = open(output_path, mode="w")
    fieldnames = ["scene_id","im_id","obj_id","score","R","t","time"]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    for r in results:
        mat = r[pose_key].copy()
        mat[:3, 3] = mat[:3, 3] * 1000.0

        score = r[score_key] if score_key in r else 1
        time = r[time_key] if time_key in r else -1

        csv_writer.writerow({
            "scene_id": r['scene_id'],
            "im_id": r['im_id'],
            "obj_id": r['obj_id'],
            "score": score,
            "R": " ".join([str(_) for _  in mat[:3, :3].flatten()]),
            "t": " ".join([str(_) for _  in mat[:3, 3].flatten()]),
            "time": time,
        })

    print("BOP logs saved to:", output_path)
    csv_file.close()

    if run_eval_script:
        print("Executing the BOP evaluation script in the background")
        os.system("cd %s; PYTHONPATH='/' python scripts/eval_bop19.py --renderer_type=cpp --result_filenames=%s" % (BOP_TOOLKIT_PATH, output_filename))

def readResultsBop(path):
    df = pd.read_csv(path)
    results = []
    for i, r in df.iterrows():
        pose = np.eye(4)
        R = np.asarray([float(_) for _ in r['R'].split(" ")]).reshape((3, 3))
        t = np.asarray([float(_) for _ in r['t'].split(" ")])
        pose[:3, :3] = R
        pose[:3, 3] = t

        results.append({
            "obj_id": int(r['obj_id']),
            "scene_id": int(r['scene_id']),
            "im_id": int(r['im_id']),
            "score": float(r['score']),
            "time": float(r['time']),
            "pose": pose
        })
    
    return results

