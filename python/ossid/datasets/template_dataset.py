import os
import pickle
import cv2
import imageio
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R


from ossid.utils import normalizeImage

from ossid.config import OSSID_DATA_ROOT

class TemplateDataset():
    '''
    A class providing utilities for loading templates from rendering grids
    '''
    def __init__(self, grid_root, obj_ids, obj_id_offset=0, preload = False, use_provided_template=False):
        self.grid_root = grid_root
        self.obj_ids = obj_ids
        self.obj_id_offset = obj_id_offset
        self.preload = preload

        # Whether to load the templates rendered by ourself or provided by DTOID authors
        self.use_provided_template = use_provided_template
        if self.use_provided_template:
            self.grid_root = os.path.join(OSSID_DATA_ROOT, "templates_LMO_DTOID")
            print("TemplateDataset: using provided templates from", self.grid_root)
            self.obj_id_offset = 0
            pose_file = os.path.join(self.grid_root, "hinterstoisser_01/poses.txt")
            poses = pd.read_csv(pose_file, sep=" ", header=None).values
            self.grid_poses = poses.reshape((-1, 4, 4))

            self.view_ids = list(range(len(self.grid_poses)))

            # Convert the pose matrices to quaternions
            self.grid_rots = self.grid_poses[:, :3, :3]
            self.grid_quats = R.from_matrix(self.grid_rots).as_quat()
            self.template_z_values = self.grid_poses[:, 2, 3]
        
        else:
            # Load the relative transformation for each renders
            vid2rot_path = os.path.join(self.grid_root, "vid2rot.pkl")
            self.vid2rot = pickle.load(open(vid2rot_path, "rb"))
            self.view_ids = list(self.vid2rot.keys())
            self.view_ids.sort()

            # Extract the rotation for each view and convert them into quaternions
            self.grid_rots = np.stack([v for k, v in self.vid2rot.items()], axis=0)
            self.grid_quats = R.from_matrix(self.grid_rots).as_quat()

        # The cache of all the templates of all objects
        self.template_cache = {}
        if preload:
            print("TemplateDataset: Preloading all templates for all objects")
            # Pre-load all templates for all objects
            for obj_id in self.obj_ids:
                self.template_cache[obj_id] = (self.getTemplatesAll(obj_id))

    def getTemplate(self, obj_id, view_id):
        view_id = int(view_id)
        if self.preload and obj_id in self.template_cache:
            all_img, all_xyz, all_mask = self.template_cache[obj_id]
            return all_img[view_id], all_xyz[view_id], all_mask[view_id]
        else:
            obj_id = int(obj_id)
            if self.use_provided_template:
                template_folder = os.path.join(self.grid_root, "hinterstoisser_%02d" % obj_id)
                color_path = os.path.join(template_folder, "%06d_a.png" % view_id)
                depth_path = os.path.join(template_folder, "%06d_d.png" % view_id)
                mask_path = os.path.join(template_folder, "%06d_m.png" % view_id)

                img = cv2.imread(color_path)[:, :, ::-1]
                # (124, 124, 3) in np.uint8. Do not know how to recover it to XYZ
                depth = cv2.imread(depth_path)
                # TODO: Figure out the camera intrinsics matrix if needed
                xyz = depth
                mask = cv2.imread(mask_path)[:, :, 0].astype(np.float32) / 255.0
            else:
                # The offset is only used when actually loading data from the disk
                obj_id = obj_id + self.obj_id_offset

                render_folder = os.path.join(self.grid_root, "%06d" % obj_id)
                color_path = os.path.join(render_folder, "%04d_color.png" % view_id)
                xyz_path = os.path.join(render_folder, "%04d_xyz.npy" % view_id)
                mask_path = os.path.join(render_folder, "%04d_mask.npy" % view_id)
                img = imageio.imread(color_path)
                xyz = np.load(xyz_path)
                mask = np.load(mask_path)

        # Processing for PyTroch
        img = img.transpose(2, 0, 1)
        img = normalizeImage(img).astype(np.float32)
        mask = mask[None].astype(np.float32)
        xyz = xyz.transpose(2, 0, 1).astype(np.float32)

        return img, xyz, mask
            
    def getTemplatesAll(self, obj_id):
        if self.preload and obj_id in self.template_cache:
            return self.template_cache[obj_id]
        else:
            all_img = []
            all_xyz = []
            all_mask = []

            for vid in self.view_ids:
                img, xyz, mask = self.getTemplate(obj_id, vid)
                all_img.append(img)
                all_xyz.append(xyz)
                all_mask.append(mask)
            
            all_img = np.stack(all_img, 0)
            all_xyz = np.stack(all_xyz, 0)
            all_mask = np.stack(all_mask, 0)

            return all_img, all_xyz, all_mask