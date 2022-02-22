import cv2
import numpy as np
import torch
import time

from ossid.utils import K2meta, to_np

from zephyr.utils import projectPointsUv

def networkInference(model, dataset, data, return_time = False):
    # Convert the data to pytorch data format
    scoring_data = {}
    img_blur = cv2.GaussianBlur(data['img'], (5,5), 0)
    scoring_data['img'] = torch.from_numpy(img_blur/255.)
    scoring_data['depth'] = torch.from_numpy(data['depth'])
    scoring_data['transforms'] = torch.from_numpy(data['pose_hypos'])
    scoring_data['meta_data'] = K2meta(data['cam_K'])
    scoring_data['model_points'] = torch.from_numpy(data['model_points'])
    scoring_data['model_colors'] = torch.from_numpy(data['model_colors'])
    scoring_data['model_normals'] = torch.from_numpy(data['model_normals'])
    
    # If we have GT error, store it. Otherwise use a dummy one
    if "pp_err" not in data:
        scoring_data['pp_err'] = torch.zeros(len(data['pose_hypos']))
    else:
        scoring_data['pp_err'] = data['pp_err']
    
    with torch.no_grad():
        t1 = time.time()
        # Pre-process the data
        point_x, uv_original = dataset.getPointNetData(scoring_data, return_uv_original = True)

        # Network inference
        pred_score = model({"point_x": point_x.to(model.device)})
        pred_score = to_np(pred_score)
        t2 = time.time()
        inference_time = t2 - t1

    selected_poses = to_np(scoring_data['transforms'])
    selected_errs = scoring_data['pp_err']
    
    # Note that some hypotheses are excluded beforehand if it violates free-space violation too much
    # Therefore, the pp_err may be changed to a different size. 
    if return_time:
        return selected_poses, pred_score, selected_errs, uv_original, inference_time
    else:
        return selected_poses, pred_score, selected_errs, uv_original

def filterHypoByMask(model_points, meta_data, pose_hypos, mask, th = 0.5):
    '''
    model_points: float-like, np.ndarray of shape (N, 3), in meters 
    meta_data: dict, contains camera_fx camera_fy camera_cx camera_cy
    pose_hypos: float-like, np.ndarray of shape (M, 4, 4)
    mask: np.ndarray of shape (h, w), in {0, 1}
    th: threshold of keeping hypotheses - hepy pose must project more than this protion of points within the mask
    '''
    # Project the model points down to the image space
    uv = projectPointsUv(pose_hypos, model_points, meta_data)

    # points that are out of boundaries of the image
    invalid_proj = (uv[:,:,1] >= mask.shape[0]) | (uv[:,:,1] < 0) | \
                   (uv[:,:,0] >= mask.shape[1]) | (uv[:,:,0] < 0)
    uv[invalid_proj] = 0

    inmask = mask[uv[:, :, 1], uv[:, :, 0]] # rightward is postive x and downward is positive y
    inmask[invalid_proj] = 0
    inmask_count = inmask.sum(-1)
    inmask_ratio = inmask_count / model_points.shape[0]
    
    kept_mask = inmask_ratio > th
    return kept_mask