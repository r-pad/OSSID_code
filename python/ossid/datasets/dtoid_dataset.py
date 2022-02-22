import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from scipy.spatial.transform import Rotation as R

import torch
import imageio
from torch.utils.data import Dataset, DataLoader

from ossid.datasets.render_dataset import loadHdf5
from ossid.datasets.template_dataset import TemplateDataset
from ossid.datasets.utils import collate_fn, getSampler
from ossid.utils import normalizeImage, quatAngularDiffBatch, heatmapGaussain
from ossid.utils.augmentation import augmentDepthMap
from ossid.utils.data import processData

EXCLUDE_OID = [410, 428]

def getDataloaders(cfg):
    # Load meta information
    obj2fnames_path = os.path.join(cfg.dataset.dataset_root, "object2files.json")
    obj2fnames = json.load(open(obj2fnames_path, "r"))
    object_ids = list(obj2fnames.keys())

    # Exclude the objects that have problems
    object_ids = [_ for _ in object_ids if int(_) not in EXCLUDE_OID]

    # Convert the file names to file paths
    obj2paths = {}
    for obj_id in object_ids:
        obj2paths[obj_id] = [os.path.join(cfg.dataset.dataset_root, "%s.hdf5" % fn) for fn in obj2fnames[obj_id]]

    # Split the train, valid objects
    train_obj_ids = [_ for _ in object_ids if int(_)%4 != 0]
    valid_obj_ids = [_ for _ in object_ids if int(_)%4 == 0]
    test_obj_ids = valid_obj_ids

    # For debugging, only a small porition of the data will be trained and validated on
    if cfg.debug:
        train_obj_ids = train_obj_ids[:2]
        valid_obj_ids, test_obj_ids = valid_obj_ids[:2], valid_obj_ids[2:4]
        for k, v in obj2paths.items():
            obj2paths[k] = v[:10]
    
    print("train obj:", len(train_obj_ids), "valid obj:", len(valid_obj_ids), "test obj:", len(test_obj_ids))

    # Further split the valid seen and valid unseen images
    train_set, valseen_set, valunseen_set, test_set = {}, {}, {}, {}

    for obj_id in train_obj_ids:
        obj_paths = obj2paths[obj_id]
        train_set[obj_id] = obj_paths[ : len(obj_paths)//4*3]
        valseen_set[obj_id] = obj_paths[len(obj_paths)//4*3 : ]

    for obj_id in valid_obj_ids:
        valunseen_set[obj_id] = obj2paths[obj_id]

    for obj_id in test_obj_ids:
        test_set[obj_id] = obj2paths[obj_id]
        # test_set[obj_id] = obj2paths[obj_id][::10]

    # Initialize the DataLoader class
    train_dataset = DtoidDataset("train", train_set, cfg.dataset)
    train_loader = DataLoader(
        train_dataset,num_workers=cfg.train.num_workers, collate_fn = collate_fn, pin_memory=True,  
        **(getSampler(train_dataset, cfg.train.batch_size, shuffle=True, ttt_sampling=cfg.dataset.ttt_sampling))
    )
    valseen_dataset = DtoidDataset("valid", valseen_set, cfg.dataset)
    valseen_loader = DataLoader(
        valseen_dataset, num_workers=cfg.train.num_workers, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(valseen_dataset, cfg.train.batch_size, shuffle=cfg.train.val_shuffle, ttt_sampling=cfg.dataset.ttt_sampling))
    )
    valunseen_dataset = DtoidDataset("valid", valunseen_set, cfg.dataset)
    valunseen_loader = DataLoader(
        valunseen_dataset, num_workers=cfg.train.num_workers, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(valunseen_dataset, cfg.train.batch_size, shuffle=cfg.train.val_shuffle, ttt_sampling=cfg.dataset.ttt_sampling))
    )
    # Batch size will be 1 for testing loader
    test_dataset = DtoidDataset("test", test_set, cfg.dataset)
    test_loader = DataLoader(
        test_dataset, num_workers=1, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(test_dataset, 1, shuffle=cfg.train.val_shuffle, ttt_sampling=cfg.dataset.ttt_sampling))
    )

    print("train imgs:", len(train_loader.dataset), \
          "valseen imgs:", len(valseen_loader.dataset), \
          "valunseen imgs:", len(valunseen_loader.dataset), \
          "test imgs:", len(test_loader.dataset)
        )

    return train_loader, [valunseen_loader, valseen_loader], test_loader

class DtoidDataset(Dataset):
    def __init__(self, dataset_mode, obj2paths, cfg):
        self.dataset_mode = dataset_mode
        self.obj2paths = obj2paths

        self.dataset_root = cfg.dataset_root
        self.grid_root = cfg.grid_root
        self.augment_depth = cfg.augment_depth
        self.homographic_warp = cfg.homographic_warp
        self.heatmap_scale = cfg.heatmap_shorter_length / float(cfg.shorter_length)
        self.cfg = cfg

        self.obj_ids = list(obj2paths.keys())

        # Flatten the datapoints
        self.datapoints = []
        for obj_id, paths in self.obj2paths.items():
            self.datapoints += [(obj_id, path) for path in paths]

        self.template_dataset = TemplateDataset(
            self.grid_root, self.obj_ids, preload=self.dataset_mode in ['test']
        )
        
    def loadData(self, path, obj_id, homographic_warp=False, augment_depth=False):
        # Load the data
        data = loadHdf5(path)
        img = data['colors']
        depth = data['depth']
        segcolormap = data['segcolormap']
        segmap = data['segmap']
        cam_K = np.asarray(data['campose'][0]['cam_K']).reshape((3,3))

        if augment_depth:
            depth, _  = augmentDepthMap(depth, data['normals'])

        # get the corresponding instance mask
        inst = next(_ for _ in segcolormap if _['category_id'] == obj_id)
        cate_id = int(inst['category_id'])
        inst_id = int(inst['idx'])
        cate_mask_channel = int(inst['channel_class'])
        inst_mask_channel = int(inst['channel_instance'])

        mask = np.logical_and(segmap[:, :, cate_mask_channel] == cate_id, segmap[:, :, inst_mask_channel] == inst_id)
        mask = mask.astype(np.float32)

        out = processData(
            img, mask, depth, cam_K, 
            crop=False, zoom_factor=2.0, crop_shift=False, 
            homographic_warp = homographic_warp, 
            homographic_down_factor = self.cfg.homographic_down_factor,
            homographic_random_kpt = self.cfg.homographic_random_kpt,
            keep_aspect_ratio=self.cfg.keep_aspect_ratio, 
            warp_3d=self.cfg.homographic_warp_3d,
            shorter_length = self.cfg.shorter_length, 
        )

        out['objects'] = data['objects']

        return out
        
    def __len__(self):
        return len(self.datapoints)
    
    def __getitem__(self, idx):
        # Load the query images
        obj_id, path = self.datapoints[idx]
        im_id  = int(os.path.basename(path).split(".")[0])
        data = self.loadData(path, obj_id, homographic_warp=self.homographic_warp, augment_depth=self.augment_depth)
        img, mask, xyz = data['img'], data['mask'], data['xyz']

        # From mask get bounding box
        # Will be in (x1, y1, x2, y2) format, where x is rightward and y is downward
        h, w = mask.shape[1:3] # mask in shape (1, h, w)
        try:
            mask_pixels = np.stack(mask[0].nonzero(), axis=1)
            y1, x1 = mask_pixels.min(0)
            y2, x2 = mask_pixels.max(0)
        except:
            plt.imshow(img.transpose(1, 2, 0))
            plt.show()
            plt.imshow(mask[0])
            plt.show()
            raise
        # x1, x2 = float(x1)/w, float(x2)/w
        # y1, y2 = float(y1)/h, float(y2)/h

        # The last 1 means it is positive and pad an extra dimension
        bbox_gt = np.asarray([[x1, y1, x2, y2, 1]]) 

        # Get the center of the bounding box and convert it to a GT Gaussian heatmap
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        heatmap = heatmapGaussain(
            h*self.heatmap_scale, w*self.heatmap_scale, 
            cx*self.heatmap_scale, cy*self.heatmap_scale, 
            sigma=np.sqrt(1.5)
            # sigma=np.sqrt(self.cfg.heatmap_var)
        )
        heatmap = heatmap[None]

        # Randomly Sample one global template
        gvid = random.choice(self.template_dataset.view_ids)
        gimg, gxyz, gmask = self.template_dataset.getTemplate(obj_id, gvid)

        if self.dataset_mode in ["train"]:
            # return one local template during training time
            # Get the template with closest rotation to the gt rotation
            gt_rot = [_['obj2cam'][:3, :3] for _ in data['objects'] if _['obj_id'] == int(obj_id)]
            gt_rot = gt_rot[0][:3, :3]
            gt_quat = R.from_matrix(gt_rot).as_quat()
            angular_diff = quatAngularDiffBatch(self.template_dataset.grid_quats, gt_quat[None])
            closest_view_id = angular_diff.reshape(-1).argmin()
            # print(angular_diff[closest_view_id])

            lvid = closest_view_id
            limg, lxyz, lmask = self.template_dataset.getTemplate(obj_id, lvid)
        elif self.dataset_mode in ['valid']:
            # Return one random local template for validation
            lvid = random.choice(self.template_dataset.view_ids)
            limg, lxyz, lmask = self.template_dataset.getTemplate(obj_id, lvid)
        elif self.dataset_mode in ['test']:
            # Return all templates for this object
            template_img, template_xyz, template_mask = self.template_dataset.getTemplatesAll(obj_id)
            lvids = random.choices(list(range(len(template_img))), k=160)
            limg = template_img[lvids]
            lxyz = template_xyz[lvids]
            lmask = template_mask[lvids]
        else:
            raise Exception("Unknown dataset_mode:", self.dataset_mode)

        out = {
            "img": img, "xyz": xyz, "mask": mask,
            "gimg": gimg, "gxyz": gxyz, "gmask": gmask,
            "limg": limg, "lxyz": lxyz, "lmask": lmask,
            "bbox_gt": bbox_gt, "heatmap": heatmap,
            "obj_id": int(obj_id), 
            "scene_id": 1, 
            "im_id": im_id, 
        }

        return out