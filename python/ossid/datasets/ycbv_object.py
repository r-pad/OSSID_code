import os
import numpy as np
from object_pose_utils.datasets.ycb_grid import load_grid_data
from oriented_features.feature_grid import loadGridKeypoints, getModelPointsCloud
from oriented_features.feature_pyramid import MultiscaleGrid
from oriented_features.sift import Sift


class YcbvObject():
    def __init__(
            self, object_id, classes,
            dataset_root='/datasets/ycb/YCB_Video_Dataset', scales=[11],
            grid_file = "/home/qiaog/pose-est/feature_based_pose/scripts/verts_grid_0.npy"
        ):
        self.object_id = object_id
        self.dataset_root = dataset_root
        self.scales = scales
        self.classes = classes

        grid_indices = np.load('/home/qiaog/pose-est/feature_based_pose/scripts/verts_grid_0.npy')
        grid_images, grid_depths, grid_masks, grid_quats, grid_metas \
            = load_grid_data(self.dataset_root, grid_indices, object_id, return_meta=True)
        keypoints_all, pc_normal_color_all, kpt_idx_all = loadGridKeypoints(
            object_id, grid_indices, classes, leaf_size = 7
            )

        points = [_[:, :3] for _ in pc_normal_color_all]
        normals = [_[:, 3:6] for _ in pc_normal_color_all]

        grid = MultiscaleGrid(grid_images, grid_depths, grid_masks, grid_quats,
                              keypoints_all, points, normals, kpt_idx_all, scales=scales,
                              featurize_class=Sift, id=object_id, path_root = os.path.join(dataset_root, "./grid_data")
                              )

        grid_features = np.concatenate([grid.features[s] for s in scales], axis=0).astype('float32').copy(order='C')
        grid_clouds = np.concatenate([grid.clouds[s] for s in scales], axis=0)
        grid_frames = np.concatenate([grid.frames[s] for s in scales], axis=0)
        grid_kpt_ids = np.concatenate([grid.indices[s] for s in scales], axis=0)
        grid_harris_scores = np.concatenate([grid.harris_scores[s] for s in scales], axis=0)
        grid_view_ids = np.concatenate([grid.view_ids[s] for s in scales], axis=0)
        most_straight_features = np.concatenate([grid.most_straight_features[s] for s in scales], axis=0)
        view_kpt_features = np.concatenate([grid.view_kpt_features[s] for s in scales], axis=0)

        model_pts, model_norms, model_colors = getModelPointsCloud(
            object_id, self.classes, grid_images, keypoints_all, kpt_idx_all, leaf_size=7
        )

        self.grid = grid
        self.model_pts = model_pts
        self.model_norms = model_norms
        self.model_colors = model_colors
        self.grid_quats = np.asarray(grid_quats)
        self.grid_poses = np.asarray([_["poses"][:, :, 0] for _ in grid_metas])
        self.grid_features = grid_features
        self.grid_kpt_ids = grid_kpt_ids
        self.grid_view_ids = grid_view_ids
        self.grid_images = grid_images

        '''Compute most straight view features for each model point'''
        self.kpt_grid_cos_mat = kptProjGridCos(model_pts, model_norms, grid_metas, keypoints_all, kpt_idx_all)
        self.most_straight_features = getMostStraightFeatures(self.kpt_grid_cos_mat, view_kpt_features)

def kptProjGridCos(model_pts, model_norms, grid_metas, keypoints_all, kpt_idx_all):
    n_grids = len(grid_metas)
    n_kpts = model_pts.shape[0]
    cos_mat = np.ones((n_kpts, n_grids)) * -1

    for i_grid in range(n_grids):
        mat = grid_metas[i_grid]['poses'][:, :, 0]
        proj_kpts = keypoints_all[i_grid].astype(int)
        proj_kpts_idx = kpt_idx_all[i_grid]
        trans_norms = np.einsum('jk,mk->mj', mat[:3,:3], model_norms)
        trans_pts = np.einsum('jk,mk->mj', mat[:3, :3], model_pts) + mat[:3, 3].reshape((-1, 3))
        cam_norm_cos = (-trans_pts * trans_norms).sum(-1) / (np.linalg.norm(trans_pts, axis=-1) * np.linalg.norm(trans_norms, axis=-1))
        proj_cam_norm_cos = cam_norm_cos[proj_kpts_idx]
        cos_mat[proj_kpts_idx, np.ones(len(proj_kpts), dtype=int) * i_grid] = proj_cam_norm_cos
    return cos_mat

def getMostStraightFeatures(kpt_grid_cos_mat, view_kpt_features):
    most_straight_features = []
    kpt_straight_view_idx = (-kpt_grid_cos_mat).argsort(axis=1)
    for i_kpt in range(kpt_grid_cos_mat.shape[0]):
        for i_view in kpt_straight_view_idx[i_kpt]:
            if i_kpt in view_kpt_features[i_view]:
                most_straight_features.append(view_kpt_features[i_view][i_kpt])
                break
    most_straight_features = np.asarray(most_straight_features)
    return most_straight_features
