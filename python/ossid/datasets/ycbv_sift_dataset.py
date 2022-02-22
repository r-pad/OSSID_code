import numpy as np
from numpy.ma.extras import mask_cols
from oriented_features.feature_graph.datasets.fewshot_bop_dataset import cropMask
import torch
import cv2
import faiss
import scipy
from torch.utils.data import Dataset, DataLoader

from oriented_features.sift import Sift

from object_pose_utils.datasets.ycb_dataset import YcbDataset as YCBDataset
from object_pose_utils.datasets.pose_dataset import OutputTypes as otypes

from oriented_features.sift import Sift

from ossid.datasets.ycbv_object import YcbvObject
from ossid.utils import quatAngularDiffBatch, kpts2cloud, torch2Img, to_np, meta2K
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

def getDataloaders(cfg):
    # For fast debug
    all_objects = np.arange(1, 22)
    if cfg.debug:
        seen_objects = [1]
        unseen_objects = [2]
    else:
        if cfg.dataset.valobj == "even":
            seen_objects = [i for i in all_objects if i%2 == 1]
            unseen_objects = [i for i in all_objects if i%2 == 0]
        elif cfg.dataset.valobj == "odd":
            seen_objects = [i for i in all_objects if i%2 == 0]
            unseen_objects = [i for i in all_objects if i%2 == 1]
        else:
            raise Exception("Unknown parameter for valobj: %s" % cfg.dataset.valobj)

    train_dataset = YcbvSiftDataset("train", seen_objects, cfg)
    valid_seen_dataset = YcbvSiftDataset("valid", seen_objects, cfg)
    valid_unseen_dataset = YcbvSiftDataset("valid", unseen_objects, cfg)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers,
    )
    valid_seen_loader = DataLoader(
        valid_seen_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.val_shuffle, num_workers=cfg.train.num_workers,
    )
    valid_unseen_loader = DataLoader(
        valid_unseen_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.val_shuffle, num_workers=cfg.train.num_workers,
    )

    print("train_dataset:", len(train_dataset))
    print("valid_seen_dataset:", len(valid_seen_dataset))
    print("valid_unseen_dataset:", len(valid_unseen_dataset))

    return train_loader, [valid_seen_loader, valid_unseen_loader], None

class YcbvSiftDataset(Dataset):
    def __init__(
            self, dataset_mode, object_ids, cfg
            ):
        '''Initialize ycbv dataset'''
        self.dataset_mode = dataset_mode
        self.object_ids = object_ids
        self.dataset_root = cfg.dataset.dataset_root
        self.grid_file = cfg.dataset.grid_file
        self.skip = cfg.dataset.skip
        # self.n_kpts = cfg.dataset.n_kpts
        self.n_kpts_obs = cfg.dataset.n_kpts_obs
        self.n_kpts_model = cfg.dataset.n_kpts_model
        self.dist_thresh = cfg.dataset.dist_thresh
        self.scales = list(cfg.dataset.scales)
        self.detect = cfg.dataset.detect
        self.oracle_model_feats = cfg.dataset.oracle_model_feats
        self.oracle_model_view = cfg.dataset.oracle_model_view
        self.oracle_obs_kpts = cfg.dataset.oracle_obs_kpts
        self.obs_kpt_dim = cfg.dataset.obs_kpt_dim
        self.crop = cfg.dataset.crop

        output_format = [otypes.OBJECT_LABEL,
                         otypes.TRANSFORM_MATRIX,
                         otypes.IMAGE,
                         otypes.DEPTH_IMAGE,
                         otypes.SCENE_IM_ID,
                         ]

        ycb_dataset = YCBDataset(self.dataset_root, mode=self.dataset_mode,
                                 object_list = self.object_ids,
                                 output_data = output_format,
                                 image_size = [640, 480],
                                 )
        self.ycb_dataset = ycb_dataset

        '''Initialize feature pyramid for models'''
        self.objects = {}
        for object_id in self.object_ids:
            self.objects[object_id] = YcbvObject(object_id, ycb_dataset.classes, self.dataset_root, self.scales, self.grid_file)

        self.featurizer = Sift(11, 11)

    def __getitem__(self, idx):
        idx = idx * self.skip
        '''Get the scene data'''
        obj_id, mat, img, depth, (scene_id, im_id) = self.ycb_dataset[idx]
        obj = self.objects[obj_id.item()]
        model_pts = obj.model_pts
        model_norms = obj.model_norms
        most_straight_features = obj.most_straight_features

        meta_data = self.ycb_dataset.getMetaData(idx, camera_matrix=True, mask=True)
        mask = meta_data['mask_label']
        img = torch2Img(img, normalized=False).astype(np.uint8)
        mat = to_np(mat)
        depth = to_np(depth).astype(float)
        depth /= meta_data['camera_scale']

        if self.obs_kpt_dim == 3:
            feature_mask = depth > 1e-6
        else:
            feature_mask = np.ones_like(mask).astype(bool)

        # Get featurization only at a cropped region centered at the GT mask
        if self.crop:
            crop_mask = mask2CropBox(mask, zoom_factor=2.0, shift = self.dataset_mode == "train")
            feature_mask = np.logical_and(feature_mask, crop_mask)

        keypoints, features = self.featurizer(img,
            detect_keypoints=self.detect,
            mask = feature_mask
            )
        obs_kpts = cv2.KeyPoint_convert(keypoints)
        obs_feats = features

        # Compute the obs feats scores using mask
        gt_obs_kpt_score = mask[obs_kpts[:, 1].astype(int), obs_kpts[:, 0].astype(int)]

        # Sub-sample the keypoints on the observation side
        # Observed keypoints is over 3000
        if len(obs_kpts) > self.n_kpts_obs:
            if self.oracle_obs_kpts:
                obs_idxs = gt_obs_kpt_score.nonzero()[0]
                if len(obs_idxs) < self.n_kpts_obs:
                    other_idxs = (1 - gt_obs_kpt_score).nonzero()[0]
                    other_idxs = np.random.permutation(other_idxs)[:self.n_kpts_obs - len(obs_idxs)]
                    obs_idxs = np.concatenate([obs_idxs, other_idxs])
                else:
                    obs_idxs = obs_idxs[:self.n_kpts_obs]
            else:
                obs_idxs = np.random.permutation(len(obs_kpts))[:self.n_kpts_obs]
            obs_kpts = obs_kpts[obs_idxs]
            obs_feats = obs_feats[obs_idxs]
            gt_obs_kpt_score = gt_obs_kpt_score[obs_idxs]
        else:
            # Pad the observed keypoints and features
            to_pad = self.n_kpts_obs - len(obs_kpts)
            obs_kpts = np.pad(obs_kpts, ((0, to_pad), (0,0)), mode="constant", constant_values=0)
            obs_feats = np.pad(obs_feats, ((0, to_pad), (0,0)), mode="constant", constant_values=0)
            gt_obs_kpt_score = np.pad(gt_obs_kpt_score, ((0, to_pad)), mode="constant", constant_values=0)

        # If oracle_model_view, get the model keypoints and features from the view closest to the GT rotation
        if self.oracle_model_view:
            obs_rot = R.from_matrix(mat[:3, :3])
            obs_quat = obs_rot.as_quat()
            rot_dist = quatAngularDiffBatch(obs_quat[None], obj.grid_quats)[0]
            idx_closest_view = rot_dist.argmin()
            kpt_ids_closest = obj.grid_kpt_ids[obj.grid_view_ids == idx_closest_view]
            feats_closest = obj.grid_features[obj.grid_view_ids == idx_closest_view]

            model_kpts = model_pts[kpt_ids_closest]
            model_kpt_norms = model_norms[kpt_ids_closest]
            model_feats = feats_closest
        else:
            model_kpts = model_pts.copy()
            model_kpt_norms = model_norms.copy()
            model_feats = most_straight_features

        # Sample the points on the model side
        # model points can be less
        model_idxs = np.arange(len(model_kpts))
        if len(model_kpts) < self.n_kpts_model:
            n_valid_kpts = len(model_kpts)
            model_kpts = np.vstack([model_kpts, np.zeros((self.n_kpts_model - model_kpts.shape[0], model_kpts.shape[1]))])
            model_kpt_norms = np.vstack([model_kpt_norms, np.zeros((self.n_kpts_model - model_kpt_norms.shape[0], model_kpt_norms.shape[1]))])
            model_feats = np.vstack([model_feats, np.zeros((self.n_kpts_model - model_feats.shape[0], model_feats.shape[1]))])
        else:
            n_valid_kpts = self.n_kpts_model
            model_idxs = np.random.permutation(len(model_kpts))[:self.n_kpts_model]
            model_kpts = model_kpts[model_idxs]
            model_kpt_norms = model_kpt_norms[model_idxs]
            model_feats = model_feats[model_idxs]

        proj_uv, proj_idx = projectModelPoint(
            model_kpts[:n_valid_kpts], model_kpt_norms[:n_valid_kpts], meta_data, mat, img, mask
        )

        # Assign match by projection and Hungarian algorithm
        row_ind, col_ind, match_dist = assignMatches(obs_kpts[gt_obs_kpt_score], proj_uv, self.dist_thresh)
        gt_obs_kpt_idx = np.nonzero(gt_obs_kpt_score)[0]
        gt_sim_mat = np.zeros((len(obs_kpts), len(model_kpts)))
        gt_sim_mat[gt_obs_kpt_idx[row_ind], proj_idx[col_ind]] = 1

        # DEBUG: Use the observed features as the oracle model features
        if self.oracle_model_feats:
            # obs_feats[gt_obs_kpt_idx[row_ind]] = model_feats[proj_idx[col_ind]]
            model_feats[proj_idx[col_ind]] = obs_feats[gt_obs_kpt_idx[row_ind]]

        # Assign those non-match to trash bin
        gt_sim_mat = np.hstack([gt_sim_mat, gt_sim_mat.sum(1).reshape((-1, 1)) == 0])
        gt_sim_mat = np.vstack([gt_sim_mat, gt_sim_mat.sum(0) == 0])
        gt_sim_mat[-1, -1] = 0

        data = {
            "obs_kpts": torch.from_numpy(obs_kpts).float(), # (n_batch, n_obs_kpt, 2)
            "obs_feats": torch.from_numpy(obs_feats).float(), # (n_batch, n_obs_kpt, dim_feat)
            "model_kpts": torch.from_numpy(model_kpts).float(), # (n_batch, n_model_kpt, dim_pos)
            "model_feats": torch.from_numpy(model_feats).float(), # (n_batch, n_model_kpt, dim_feat)
            "gt_sim_mat": torch.from_numpy(gt_sim_mat).float(), # (n_batch, n_obs_kpt+1, n_model_kpt+1)
            "gt_obs_kpt_score": torch.from_numpy(gt_obs_kpt_score).float(), # (n_batch, n_obs_kpt)
            "img": torch.from_numpy(img),
            "mask": torch.from_numpy(mask),
            # 'depth': depth,
            # "model_img": obj.grid_images[idx_closest_view],
            # "model_norms": model_norms[model_idxs],
            # "meta_data": meta_data,
            # "mat": mat
        }

        '''Lift the 2D observed kpts to 3D'''
        if self.obs_kpt_dim == 3:
            cam_K = meta2K(meta_data)
            obs_kpts_3d = kpts2cloud(obs_kpts, depth, cam_K)
            data['obs_kpts_3d'] = torch.from_numpy(obs_kpts_3d).float(), # (n_batch, n_obs_kpt, 3)

        return data

    def __len__(self):
        return len(self.ycb_dataset) // self.skip

def mask2CropBox(mask, zoom_factor = 1.0, shift=False):
    h, w = mask.shape

    mask_idx = mask.nonzero()
    min_x, max_x = mask_idx[0].min(), mask_idx[0].max()
    min_y, max_y = mask_idx[1].min(), mask_idx[1].max()
    mh, mw = max_x - min_x, max_y - min_y
    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2

    r = max(max_x - min_x, max_y - min_y) // 2    
    r = int(zoom_factor * r)

    if shift:
        cx, cy = cx + int(np.random.rand() * r - r/2.0), cy + int(np.random.rand() * r - r/2.0)
        cx = int(np.clip(cx, 0.2*h, 0.8*h))
        cy = int(np.clip(cy, 0.2*w, 0.8*w))

    crop_mask = np.zeros_like(mask)
    crop_mask[
        max(cx-r, 0):min(cx+r, h), 
        max(cy-r, 0):min(cy+r, w)
        ] = 1
    return crop_mask

def assignMatches(obs_kpts, proj_uv, dist_thresh = 5):
    # Compute cross pairwise distance and run Hungarian algorithm
    dist_mat = scipy.spatial.distance.cdist(obs_kpts, proj_uv, 'euclidean')
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist_mat)

    # Filter out those matches that are farther than the threshold
    match_dist = dist_mat[row_ind, col_ind]
    row_ind, col_ind = row_ind[match_dist <= dist_thresh], col_ind[match_dist <= dist_thresh]
    match_dist = match_dist[match_dist <= dist_thresh]

    return row_ind, col_ind, match_dist

def getGTSimMat(proj_uv, proj_idx, obs_kpts, model_pts, thresh_px = 5):
    if len(proj_uv) == 0:
        return np.zeros((len(obs_kpts), len(model_pts)))
    proj_index = createSearchIndex(proj_uv.astype(np.float32))
    proj_knn_dists, proj_knn_idxs = proj_index.search(obs_kpts, k=1)
    proj_knn_dists = proj_knn_dists ** (0.5)
    proj_knn_dists = proj_knn_dists.reshape(-1)
    proj_knn_idxs = proj_knn_idxs.reshape(-1)
    match_model_idx = proj_idx[proj_knn_idxs]
    gt_sim_mat = np.zeros((len(obs_kpts), len(model_pts)))
    obs_feat_mask = np.arange(len(obs_kpts))[proj_knn_dists <= thresh_px]
    second_idx = match_model_idx[proj_knn_dists <= thresh_px]

    gt_sim_mat[obs_feat_mask, second_idx] = 1
    return gt_sim_mat

def createSearchIndex(data, gpu=None):
    if gpu is not None:
        gpu_res = faiss.StandardGpuResources()
        search_index = faiss.IndexFlatL2(data.shape[1])
        search_index = faiss.index_cpu_to_gpu(gpu_res, feature_gpu, search_index)
    else:
        search_index = faiss.IndexFlatL2(data.shape[1])
    search_index.add(data)
    return search_index

def projectModelPoint(model_pts, model_norms, meta_data, mat, img, mask=None):
    trans_pts = np.einsum('jk,mk->mj', mat[:3, :3], model_pts) + mat[:3, 3].reshape((-1, 3))
    f_cam = np.asarray([meta_data['camera_fx'], meta_data['camera_fy']])
    c_cam = np.asarray([meta_data['camera_cx'], meta_data['camera_cy']])
    trans_norms = np.einsum('jk,mk->mj', mat[:3,:3], model_norms)
    proj_pts = trans_pts[:,:2]/trans_pts[:,2:]*f_cam + c_cam
    valid_norm = (-trans_pts * trans_norms).sum(-1) > 0

    uv = proj_pts.round().astype(int)
    invalid_proj = (uv[:,1]>=img.shape[0]) + (uv[:,1]<0) \
                    + (uv[:,0]>=img.shape[1]) + (uv[:,0]< 0)
    valid_proj = np.logical_and(~invalid_proj, valid_norm)

    proj_idx = np.arange(model_pts.shape[0])
    proj_uv = uv[valid_proj]
    proj_idx = proj_idx[valid_proj]
    trans_pts = trans_pts[valid_proj]

    # if depth is not None:
    #     proj_depth = trans_pts[:, -1]
    #     depth = depth.astype(float) / meta_data['camera_scale']
    #     obs_depth = depth[proj_uv[:, 1], proj_uv[:, 0]]
    #     valid_depth = np.absolute(proj_depth - obs_depth) < 0.02
    #     valid_proj = np.logical_and(valid_proj, depth_match)

    if not mask is None:
        valid_mask = mask[proj_uv[:, 1], proj_uv[:, 0]] > 0
        proj_uv = proj_uv[valid_mask]
        proj_idx = proj_idx[valid_mask]
        trans_pts = trans_pts[valid_mask]

    return proj_uv, proj_idx
