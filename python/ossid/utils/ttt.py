import os
import pickle
import pandas as pd

def readLog(log_paths, keys = ['desc_loss', "seg_loss", "seg_IoU"]):
    dfs = []
    for log_path in log_paths:
        df = []
        result = pickle.load(open(log_path, 'rb'))
        for batch_id, log in enumerate(result['test_logs']):
            obj_id = log['obj_id']
            for step in range(len(log[keys[0]])):
                d = {
                    "obj_id": obj_id,
                    "batch_id": batch_id,
                    "step": step,
                    "log_name": os.path.basename(log_path)
                }
                for k in keys:
                    d[k] = log[k][step]
                    
                df.append(d)
                    
        df = pd.DataFrame(df)
        dfs.append(df)

    df_all = pd.concat(dfs)

    return df_all