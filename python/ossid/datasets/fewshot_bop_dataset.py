import numpy as np
import torch
import os
import random
import cv2
import copy
import argparse
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from ossid.datasets.utils import collate_fn, getSampler
from ossid.utils import kpts2cloud, meta2K, depth2xyz, TorchTimer
from ossid.utils.data import processData
from oriented_features.full_pipeline.bop_dataset import BopDataset
from oriented_features.pose_scoring_lightning.constants import YCBV_TRAIN_SCENE, YCBV_VALID_SCENE, YCBV_BOPTEST_SCENE
from torch.utils.data.sampler import SubsetRandomSampler


def getDataloaders(cfg):
    if cfg.dataset.dataset_name == "ycbv":
        return getYcbvDataloaders(cfg)
    elif cfg.dataset.dataset_name == "lmo":
        return getLmoDataloaders(cfg)
    else:
        raise NotImplementedError("Unknown dataset_name:", cfg.dataset.dataset_name)

def getLmoDataloaders(cfg):
    print("Processing LM-O dataset")
    # Manipulate the arguments used for BopDataset
    # For testing on LM-O, use the LM test set as training set
    train_args = argparse.Namespace()
    train_args.bop_root = cfg.dataset.bop_root
    train_args.dataset_name = "lm"
    train_args.split_name = "test"
    train_args.split = "test"
    train_args.split_type = None
    train_args.model_type = None
    train_args.skip = 3

    # Use the LM-O bop_test set as the testing set
    test_args = argparse.Namespace()
    test_args.bop_root = cfg.dataset.bop_root
    test_args.dataset_name = "lmo"
    test_args.split_name = "bop_test"
    test_args.split = "test"
    test_args.split_type = None
    test_args.model_type = None
    test_args.skip = 1

    # Initialize the training and testing dataset
    train_bop_dataset = BopDataset(train_args)
    trainall_bop_dataset = BopDataset(train_args) # This is used for support images during test time
    test_bop_dataset = BopDataset(test_args)

    # Split the object into seen and unseen
    # Also scene 02 is in the test sequence, remove it from the training set
    all_objects = train_bop_dataset.obj_ids
    unseen_objects = test_bop_dataset.obj_ids
    seen_objects = [_ for _ in all_objects if _ not in unseen_objects + [2]] # Object with id=2 also needs to be removed from the seen objects as its corresponding scene is removed. 
    all_scenes = list(range(1, 16))
    test_scenes = [2]
    train_scenes = [_ for _ in all_scenes if _ not in test_scenes]

    print("seen_objects:", seen_objects)
    print("unseen_objects:", unseen_objects)
    print("train_scenes:", train_scenes)
    print("test_scenes:", test_scenes)

    train_bop_dataset.targets = [_ for _ in train_bop_dataset.targets if _['obj_id'] in seen_objects]
    train_bop_dataset.targets = [_ for _ in train_bop_dataset.targets if _['scene_id'] in train_scenes]
    test_bop_dataset.targets = [_ for _ in test_bop_dataset.targets if _['obj_id'] in unseen_objects]

    # Split the train dataset into train and valseen
    valseen_bop_dataset = copy.deepcopy(train_bop_dataset)
    valseen_bop_dataset.targets = [valseen_bop_dataset.targets[i] for i in range(len(valseen_bop_dataset.targets)) if i%4 == 0]
    train_bop_dataset.targets = [train_bop_dataset.targets[i] for i in range(len(train_bop_dataset.targets)) if i%4 != 0]

    # Remove the testing scene from the support images
    trainall_bop_dataset.targets = [_ for _ in trainall_bop_dataset.targets if _['scene_id'] in train_scenes]

    train_dataset = FewshotBopDataset("train", seen_objects, train_bop_dataset, cfg)
    train_loader = DataLoader(
        train_dataset, num_workers=cfg.train.num_workers, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(train_dataset, cfg.train.batch_size, shuffle=True, ttt_sampling=cfg.dataset.ttt_sampling))
    )
    valseen_dataset = FewshotBopDataset("valid", seen_objects, valseen_bop_dataset, cfg, trainall_bop_dataset)
    valseen_loader = DataLoader(
        valseen_dataset, num_workers=cfg.train.num_workers, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(valseen_dataset, cfg.train.batch_size, shuffle=cfg.train.val_shuffle, ttt_sampling=cfg.dataset.ttt_sampling))
    )
    testunseen_dataset = FewshotBopDataset("test", unseen_objects, test_bop_dataset, cfg, trainall_bop_dataset)
    testunseen_loader = DataLoader(
        testunseen_dataset, num_workers=cfg.train.num_workers, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(testunseen_dataset, cfg.train.batch_size, shuffle=cfg.train.val_shuffle, ttt_sampling=cfg.dataset.ttt_sampling))
    )

    return train_loader, valseen_loader, [testunseen_loader]

def getYcbvDataloaders(cfg):
    # Split object into seen and unseen categories
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

    # get the training and validation datasets
    train_args = copy.deepcopy(cfg.dataset)
    # train_args.split_name = "train_real"
    # train_args.split = "train"
    # train_args.split_type = "real"

    train_bop_dataset = BopDataset(train_args)
    trainall_bop_dataset = copy.deepcopy(train_bop_dataset)
    valseen_bop_dataset = BopDataset(train_args)
    valunseen_bop_dataset = BopDataset(train_args)

    # For YCB-V in BOP, there is no provided validation set. So we split it manually
    if train_args.split_type == "real":
        train_bop_dataset.targets = [_ for _ in train_bop_dataset.targets if \
            _['obj_id'] in seen_objects and _['scene_id'] in YCBV_TRAIN_SCENE]
        valseen_bop_dataset.targets = [_ for _ in valseen_bop_dataset.targets if \
            _['obj_id'] in seen_objects and _['scene_id'] in YCBV_VALID_SCENE]
        valunseen_bop_dataset.targets = [_ for _ in valunseen_bop_dataset.targets if \
            _['obj_id'] in unseen_objects and _['scene_id'] in YCBV_VALID_SCENE]
    elif train_args.split_type == "pbr":
        # train_bop_dataset.targets = [_ for _ in train_bop_dataset.targets if \
        #     _['obj_id'] in seen_objects and _['scene_id'] not in list(range(40, 50))]
        train_bop_dataset.targets = [_ for _ in train_bop_dataset.targets if \
            _['obj_id'] in [1] and _['scene_id'] not in list(range(40, 50))]
        valseen_bop_dataset.targets = [_ for _ in valseen_bop_dataset.targets if \
            _['obj_id'] in seen_objects and _['scene_id'] in list(range(40, 50))]
        valunseen_bop_dataset.targets = [_ for _ in valunseen_bop_dataset.targets if \
            _['obj_id'] in unseen_objects and _['scene_id'] in list(range(40, 50))]
    else:
        raise Exception("Unknown train_args.split_type:", train_args.split_type)

    # Get the testing datasets
    test_args = copy.deepcopy(cfg.dataset)
    test_args.split_name = "bop_test"
    test_args.split = "test"
    test_args.split_type = None

    testseen_bop_dataset = BopDataset(test_args)
    testunseen_bop_dataset = BopDataset(test_args)
    testseen_bop_dataset.targets = [_ for _ in testseen_bop_dataset.targets if \
        _['obj_id'] in seen_objects]
    testunseen_bop_dataset.targets = [_ for _ in testunseen_bop_dataset.targets if \
        _['obj_id'] in unseen_objects]

    print("train:", len(train_bop_dataset), \
          "valseen:", len(valseen_bop_dataset), \
          "valunseen:", len(valunseen_bop_dataset), \
          "testseen:", len(testseen_bop_dataset), \
          "testunseen:", len(testunseen_bop_dataset), \
        )

    # Initialize all dataset and dataloader class
    train_dataset = FewshotBopDataset("train", seen_objects, train_bop_dataset, cfg)
    train_loader = DataLoader(
        train_dataset, num_workers=cfg.train.num_workers, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(train_dataset, cfg.train.batch_size, shuffle=True, ttt_sampling=cfg.dataset.ttt_sampling))
    )
    valseen_dataset = FewshotBopDataset("valid", seen_objects, valseen_bop_dataset, cfg, trainall_bop_dataset)
    valseen_loader = DataLoader(
        valseen_dataset, num_workers=cfg.train.num_workers, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(valseen_dataset, cfg.train.batch_size, shuffle=cfg.train.val_shuffle, ttt_sampling=cfg.dataset.ttt_sampling))
    )
    valunseen_dataset = FewshotBopDataset("valid", unseen_objects, valunseen_bop_dataset, cfg, trainall_bop_dataset)
    valunseen_loader = DataLoader(
        valunseen_dataset, num_workers=cfg.train.num_workers, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(valunseen_dataset, cfg.train.batch_size, shuffle=cfg.train.val_shuffle, ttt_sampling=cfg.dataset.ttt_sampling))
    )
    testseen_dataset = FewshotBopDataset("valid", seen_objects, testseen_bop_dataset, cfg, trainall_bop_dataset)
    testseen_loader = DataLoader(
        testseen_dataset, num_workers=cfg.train.num_workers, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(testseen_dataset, cfg.train.batch_size, shuffle=cfg.train.val_shuffle, ttt_sampling=cfg.dataset.ttt_sampling))
    )
    testunseen_dataset = FewshotBopDataset("valid", unseen_objects, testunseen_bop_dataset, cfg, trainall_bop_dataset)
    testunseen_loader = DataLoader(
        testunseen_dataset, num_workers=cfg.train.num_workers, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(testunseen_dataset, cfg.train.batch_size, shuffle=cfg.train.val_shuffle, ttt_sampling=cfg.dataset.ttt_sampling))
    )

    return train_loader, [valunseen_loader, valseen_loader], [testunseen_loader, testseen_loader]

class FewshotBopDataset(Dataset):
    def __init__(self, dataset_mode, object_ids, bop_dataset, cfg, support_bop_dataset = None):
        self.dataset_mode = dataset_mode
        self.object_ids = object_ids

        self.dataset_name = cfg.dataset.dataset_name
        self.bop_dataset = bop_dataset
        if support_bop_dataset is None:
            self.qs_same_set = True
            self.support_bop_dataset = bop_dataset
        else:
            self.qs_same_set = False
            self.support_bop_dataset = support_bop_dataset

        self.crop = cfg.dataset.crop
        self.zoom_factor = cfg.dataset.zoom_factor
        self.val_random_shift = cfg.dataset.val_random_shift
        self.k_support = cfg.dataset.k_support
        self.homographic_warp = cfg.dataset.homographic_warp
        self.cfg = cfg.dataset

        # Handle the object renders
        self.render_support = cfg.dataset.render_support
        if self.render_support:
            self.render_folder = os.path.join(self.bop_dataset.dataset_root, cfg.dataset.render_folder)
            self.grid_indices = np.load(os.path.join(self.render_folder, "verts_grid_0.npy"))
        
        # sort the query images by their object ids
        self.queries_by_oid = {}
        for oid in object_ids:
            queries_oid = [_ for _ in self.bop_dataset.targets if _['obj_id'] == oid]
            queries_oid.sort(key = lambda x: (x['scene_id'], x['im_id']))
            self.queries_by_oid[oid] = queries_oid

        # If support_bop_dataset is not provided, then the support images are from the same set as the query
        if self.qs_same_set:
            self.supports_by_oid = self.queries_by_oid
        else:
            self.supports_by_oid = {}
            for oid in object_ids:
                supports_oid = [_ for _ in self.support_bop_dataset.targets if _['obj_id'] == oid]
                supports_oid.sort(key = lambda x: (x['scene_id'], x['im_id']))
                if self.cfg.visib_fract_th > 0 and self.dataset_mode != "train":
                    supports_oid = filterTargetByVisibFract(self.support_bop_dataset, supports_oid, self.cfg.visib_fract_th)
                self.supports_by_oid[oid] = supports_oid

        self.tasks = self.pairTarget(self.queries_by_oid, self.supports_by_oid, self.object_ids)

    def pairTarget(self, queries_by_oid, supports_by_oid, object_ids):
        tasks = []

        for oid in object_ids:
            queries_oid = queries_by_oid[oid]
            supports_oid = supports_by_oid[oid]

            if self.dataset_mode == "train":
                for i in range(len(queries_oid)):
                    tasks.append({
                        "s": random.choices(supports_oid, k = self.k_support),
                        "q": queries_oid[i]
                    })
            else:
                for i in range(len(queries_oid) // 2):
                    # For each query, sample uniformly the support images from the other half of the dataset. 
                    j = i + len(queries_oid) // 2
                    interval = len(supports_oid) // 2 // self.k_support
                    tasks.append({
                        "s": [supports_oid[(i + _*interval) % len(supports_oid)] for _ in range(self.k_support)],
                        "q": queries_oid[j]
                    })
                    tasks.append({
                        "s": [supports_oid[(j + _*interval) % len(supports_oid)] for _ in range(self.k_support)],
                        "q": queries_oid[i]
                    })
                
        return tasks

    def __len__(self):
        return len(self.tasks)

    def loadData(self, data, shift = False):
        img = np.asarray(data['img'])
        mask = np.asarray(data['mask_gt_visib']) / 255.0

        depth = np.asarray(data['depth'])
        scene_meta = data['scene_meta']
        cam_K = meta2K(scene_meta)

        out = processData(
            img, mask, depth, cam_K, 
            crop = self.crop, 
            zoom_factor = self.zoom_factor, 
            crop_shift = shift, 
            homographic_warp = self.homographic_warp, 
            homographic_down_factor=self.cfg.homographic_down_factor,
            homographic_random_kpt = self.cfg.homographic_random_kpt,
            keep_aspect_ratio=self.cfg.keep_aspect_ratio,
            warp_3d=self.cfg.homographic_warp_3d,
        )

        return out

    def __getitem__(self, idx):
        t = self.tasks[idx]

        # load the query data
        # print("query:", t['q']['obj_id'], t['q']['scene_id'], t['q']['im_id'])
        q_rawdata = self.bop_dataset.getDataByIds(t['q']['obj_id'], t['q']['scene_id'], t['q']['im_id'])
        qdata = self.loadData(q_rawdata, shift = (self.dataset_mode=="train") or self.val_random_shift)
        qimg, qmask, qxyz = qdata['img'], qdata['mask'], qdata['xyz']

        simgs = []
        smasks = []
        sxyzs = []
        
        for s_id, d in enumerate(t['s']):
            if self.render_support:
                if self.dataset_mode == "train":
                    grid_id = np.random.choice(self.grid_indices)
                else:
                    grid_id = self.grid_indices[s_id]
                img, depth, mask, q, scene_meta = loadRenderData(self.render_folder, grid_id, d['obj_id'])
                s_rawdata = copy.deepcopy(d)
                s_rawdata['img'] = img
                s_rawdata['depth'] = depth
                s_rawdata['mask_gt_visib'] = mask * 255 # Unify the mask to 0/255
                s_rawdata['scene_meta'] = scene_meta
            else:
                if self.dataset_mode == "train":
                    d = random.choice(self.supports_by_oid[d['obj_id']])
                else:
                    d = random.choice(self.supports_by_oid[d['obj_id']])
                    pass
                # print("support:", d['obj_id'], d['scene_id'], d['im_id'])
                s_rawdata = self.support_bop_dataset.getDataByIds(d['obj_id'], d['scene_id'], d['im_id'])

            sdata = self.loadData(s_rawdata, shift = False)
            simg, smask, sxyz = sdata['img'], sdata['mask'], sdata['xyz']

            # simg = simg * smask

            simgs.append(simg)
            smasks.append(smask)
            sxyzs.append(sxyz)

        simg = np.concatenate(simgs, axis=0)
        smask = np.concatenate(smasks, axis=0)
        sxyz = np.concatenate(sxyzs, axis=0)

        out = {
            "simg": simg,
            "qimg": qimg,
            "sxyz": sxyz,
            "qxyz": qxyz,
            "smask": smask,
            "qmask": qmask, 
            "obj_id": t['q']['obj_id']
        }

        if self.homographic_warp:
            for k in ['kpts', 'kpts_warp', 'H', 'TR', 'Tt']:
                out['q'+k] = qdata[k]

        return out

def filterTargetByVisibFract(bop_dataset, targets, visib_fract_th):
    targets_filtered = []
    for t in targets:
        d = bop_dataset.getMetaDataByIds(t['obj_id'], t['scene_id'], t['im_id'])
        if d['visib_fract'] >= visib_fract_th:
            targets_filtered.append(t)

    if len(targets_filtered) < 6:
        print("Warning: len(targets_filtered) < 6 for obj id:", t['obj_id'])
    return targets_filtered

# A revised version of object_pose_utils.datasets.ycb_grid.load_grid_data
def loadRenderData(render_folder, grid_id, obj_id):
    filename_format = os.path.join(render_folder, "obj_%06d" % obj_id, '{:04d}-{}.{}')

    img = cv2.cvtColor(cv2.imread(filename_format.format(grid_id, 'color', 'png')), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(filename_format.format(grid_id, 'depth', 'png'), cv2.IMREAD_UNCHANGED)
    mask = np.bitwise_and((cv2.imread(filename_format.format(grid_id, 'label', 'png'))==obj_id)[:,:,0],
                            depth > 0).astype(np.uint8)
    q = np.load(filename_format.format(grid_id, 'trans', 'npy'))
    meta = loadmat(filename_format.format(grid_id, "meta", 'mat'))
    depth = depth.astype(float) / meta['factor_depth']
    scene_meta = {
        "camera_fx": 1066.778,
        "camera_fy": 1067.487,
        "camera_cx": 312.9869,
        "camera_cy": 241.3109,
    }

    return img, depth, mask, q, scene_meta
    