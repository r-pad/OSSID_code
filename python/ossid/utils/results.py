import pickle
import numpy as np
import torch
import pandas as pd

def smooth(x, N = 25):
    if type(x) is list:
        x = np.asarray(x)
    y = np.convolve(x, np.ones(N)/N, mode='valid')
    return y

def loadResult(path, iou_key = None, exp_name = None):
    results = pickle.load(open(path, 'rb'))
    
    if type(results) is dict:
        results = results['test_results']
        
    for r in results:
        if 'dtoid_iou' in r:
            if torch.is_tensor(r['dtoid_iou']):
                r['dtoid_iou'] = r['dtoid_iou'].item()
        # if 'dtoid_score' in r:
        #     r['dtoid_score'] = r['dtoid_score'][0]

    df = pd.DataFrame.from_dict(results)
    df = df.sort_values(['im_id', 'obj_id'])
    
    if iou_key is not None:
        df['iou'] = df[iou_key]
    if exp_name is not None:
        df['from'] = exp_name
    
    return df