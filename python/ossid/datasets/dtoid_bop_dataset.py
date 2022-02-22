import pickle
from networkx.algorithms.operators.unary import reverse
import numpy as np
import os
import random
import argparse
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader


from ossid.utils import meta2K, quatAngularDiffBatch, heatmapGaussain
from ossid.utils.constants import BOP_OBJECT_ID_OFFSETS
from ossid.utils.data import processData
from ossid.datasets.utils import collate_fn, getSampler, loadProcessZephyrResults
from ossid.datasets.template_dataset import TemplateDataset

from zephyr.utils.bop_dataset import BopDataset


def getDataloaders(cfg):
    # Initialize the Bop Dataset
    test_args = argparse.Namespace()
    test_args.bop_root = cfg.dataset.bop_root
    if cfg.dataset.test_dataset_name == "lmo":
        print("Test BOP dataset: the LM-O bop test set")
        test_args.dataset_name = "lmo"
        test_args.split_name = "bop_test"
        test_args.split = "test"
        test_args.split_type = None
        test_args.model_type = None
        test_args.skip = 1

        # cfg.dataset.zephyr_filter_threshold = 20
        # cfg.dataset.zephyr_result_path = "/home/qiaog/datasets/bop/zephyr_results/test_lmo_boptest_zephyr_result.pkl"
        cfg.dataset.n_classes = 15
    elif cfg.dataset.test_dataset_name == "ycbv":
        print("Test BOP dataset: the YCB-V bop test set")
        test_args.dataset_name = "ycbv"
        test_args.split_name = "bop_test"
        test_args.split = "test"
        test_args.split_type = None
        test_args.model_type = None
        test_args.skip = 1

        # cfg.dataset.zephyr_filter_threshold = 50
        # cfg.dataset.zephyr_result_path = "/home/qiaog/datasets/bop/zephyr_results/test_ycbv_boptest_zephyr_result_unseen.pkl"
        cfg.dataset.n_classes = 21
    else:
        raise Exception("Unknown cfg.dataset.test_dataset_name:", cfg.dataset.test_dataset_name)

    test_bop_dataset = BopDataset(test_args)
    unseen_objects = test_bop_dataset.obj_ids

    if cfg.dataset.train_dataset_name is None or cfg.dataset.train_dataset_name == cfg.dataset.test_dataset_name:
        print("Train BOP dataset: the %s test set" % test_args.dataset_name)
        train_bop_dataset = deepcopy(test_bop_dataset)

        # Initialize dataste for finetuning using zephyr results
        if cfg.dataset.load_zephyr_result:
            zephyr_results_train, zephyr_results_valid = loadProcessZephyrResults(cfg)
        else:
            zephyr_results_train, zephyr_results_valid = None, None

        seen_objects = unseen_objects

    elif cfg.dataset.train_dataset_name == "lm":
        print("Train BOP dataset: the LM test set")
        train_args = argparse.Namespace()
        train_args.bop_root = cfg.dataset.bop_root
        train_args.dataset_name = 'lm'
        train_args.split_name = "bop_test"
        train_args.split = "test"
        train_args.split_type = None
        train_args.model_type = None
        train_args.skip = 1

        train_bop_dataset = BopDataset(train_args)

        if cfg.dataset.load_zephyr_result:
            # Use the zephyr results
            # Then treat LM as a finetuning set
            # Only exclude the scene that are in LMO
            print("Use LM minus LMO as the finetuning, excluding LMO scene. ")
            cfg.dataset.zephyr_result_path = "/home/qiaog/datasets/bop/zephyr_results/lm_boptest_zephyr_result.pkl"
            zephyr_results_train, zephyr_results_valid = loadProcessZephyrResults(cfg)

            zephyr_results_train = [
                _ for _ in zephyr_results_train if _['scene_id'] not in [2] and _['obj_id'] in test_bop_dataset.obj_ids
            ]
            zephyr_results_valid = [
                _ for _ in zephyr_results_valid if _['scene_id'] not in [2] and _['obj_id'] in test_bop_dataset.obj_ids
            ]
        else:
            # Use the ground truth of LM
            # Then only use LM minus LMO as a domain adaptation
            # Exclude the objects that are in LMO
            print("Use LM minus LMO as the finetuning set, excluding LMO objects and scene. ")
            zephyr_results_train, zephyr_results_valid = None, None
            
            # Filter out those datapoints in LM-O dataset
            train_bop_dataset.obj_ids = [_ for _ in  train_bop_dataset.obj_ids if _ not in test_bop_dataset.obj_ids]
            train_bop_dataset.targets = [
                _ for _ in train_bop_dataset.targets if _['obj_id'] not in test_bop_dataset.obj_ids and _['scene_id'] not in [2]
            ]

        seen_objects = train_bop_dataset.obj_ids
    elif cfg.dataset.train_dataset_name == "ycbvtrain":
        print("Train BOP dataset: the YCB-V training set")
        train_args = argparse.Namespace()
        train_args.bop_root = cfg.dataset.bop_root
        train_args.dataset_name = 'ycbv'
        train_args.split_name = "train"
        train_args.split = "train"
        train_args.split_type = None
        train_args.model_type = None
        train_args.skip = 1

        train_bop_dataset = BopDataset(train_args)

        if cfg.dataset.load_zephyr_result:
            # Use the zephyr results
            # Use the YCB-V training set for finetuning
            print("Use YCB-V training set for finetuning")
            cfg.dataset.zephyr_result_path = "/home/qiaog/datasets/bop/zephyr_results/ycbv_zephyr-final-unseen_train_zephyr_result.pkl"
            zephyr_results_train, zephyr_results_valid = loadProcessZephyrResults(cfg)

            zephyr_results_train = [
                _ for _ in zephyr_results_train if _['scene_id'] not in [2] and _['obj_id'] in test_bop_dataset.obj_ids
            ]
            zephyr_results_valid = [
                _ for _ in zephyr_results_valid if _['scene_id'] not in [2] and _['obj_id'] in test_bop_dataset.obj_ids
            ]
        else:
            raise NotImplementedError

        seen_objects = train_bop_dataset.obj_ids
    else:
        raise Exception("Unknown cfg.dataset.train_dataset_name: %s" % cfg.dataset.train_dataset_name)

    # Initialize the training Dataset and Dataloader
    train_dataset = DtoidBopDataset('train', seen_objects, train_bop_dataset, cfg.dataset, zephyr_results_train)
    train_loader = DataLoader(
        train_dataset, num_workers=cfg.train.num_workers, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(train_dataset, cfg.train.batch_size, shuffle=False, ttt_sampling=cfg.dataset.ttt_sampling))
    )

    # Initialize the validation Dataset and DataLoader
    valid_dataset = DtoidBopDataset('valid', seen_objects, train_bop_dataset, cfg.dataset, zephyr_results_valid)
    valid_loader = DataLoader(
        valid_dataset, num_workers=0, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(valid_dataset, 1, shuffle=False, ttt_sampling=False))
    )

    # Initialize the testing Dataset and Dataloader
    test_dataset = DtoidBopDataset("test", unseen_objects, test_bop_dataset, cfg.dataset)
    test_loader = DataLoader(
        test_dataset, num_workers=0, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(test_dataset, 1, shuffle=False, ttt_sampling=False))
    )

    print(
        "train_dataset:", len(train_dataset), "train_loader:", len(train_loader), 
        "\nval_dataset:", len(valid_dataset), "val_loader:", len(valid_loader),
        "\ntest_dataset:", len(test_dataset), "test_loader:", len(test_loader),
    )

    # Vaildation loader will be the same as the test loader here
    return train_loader, [test_loader, valid_loader], test_loader

class DtoidBopDataset(Dataset):
    def __init__(self, dataset_mode, obj_ids, bop_dataset, cfg, zephyr_results=None):
        self.dataset_mode = dataset_mode
        self.obj_ids = obj_ids

        self.bop_dataset = deepcopy(bop_dataset)
        self.dataset_name = self.bop_dataset.dataset_name
        self.cfg = cfg
        self.heatmap_scale = cfg.heatmap_shorter_length / float(cfg.shorter_length)

        self.grid_root = self.cfg.grid_root

        self.template_dataset = TemplateDataset(
            self.grid_root, self.obj_ids, preload=self.dataset_mode in ['test'], 
            obj_id_offset = BOP_OBJECT_ID_OFFSETS[self.bop_dataset.dataset_name],
            use_provided_template=cfg.use_provided_template
        )

        if zephyr_results is not None:
            print("DtoidBopDataset: Using zephyr results")
            self.zephyr_results = {}
            for zr in zephyr_results:
                self.zephyr_results[(zr['obj_id'], zr['scene_id'], zr['im_id'])] = zr

            self.bop_dataset.targets = []
            for zr in zephyr_results:
                self.bop_dataset.targets.append({
                    "obj_id": zr['obj_id'],
                    "scene_id": zr['scene_id'],
                    "im_id": zr['im_id'],
                    "inst_count": 1,
                })
        else:
            self.zephyr_results = None

    def clearTargets(self):
        '''
        remove all datapoints
        '''
        self.bop_dataset.targets = []

    def sortTargets(self, reverse=False):
        self.bop_dataset.targets.sort(
            reverse = reverse,
            key = lambda x: (x['scene_id'], x['im_id'], x['obj_id'])
        )
        print("Sorted the testing targets", self.bop_dataset.targets[:3])

    def addTarget(self, obj_id, scene_id, im_id, mask = None, score = None):
        '''
        Add a datapoint in the BOP dataset
        '''
        self.bop_dataset.targets.append({
            "obj_id": obj_id,
            "scene_id": scene_id,
            "im_id": im_id,
            "inst_count": 1,
        })

    def updateZephyrMask(self, obj_id, scene_id, im_id, mask, score):
        '''
        Update the pseudo ground-truth in the zephyr results for better finetuning
        '''
        self.zephyr_results[(obj_id, scene_id, im_id)]['pred_mask_visib'] = mask
        self.zephyr_results[(obj_id, scene_id, im_id)]['score'] = score
    
    def __len__(self):
        return len(self.bop_dataset)

    def loadData(self, data, shift = False):
        img = np.asarray(data['img'])
        mask = np.asarray(data['mask_gt_visib']) / 255.0

        depth = np.asarray(data['depth'])
        scene_meta = data['scene_meta']
        cam_K = meta2K(scene_meta)

        out = processData(
            img, mask, depth, cam_K, 
            keep_aspect_ratio=self.cfg.keep_aspect_ratio,
            shorter_length=self.cfg.shorter_length
        )

        return out

    def __getitem__(self, idx):
        bop_data = self.bop_dataset[idx]
        obj_id = bop_data['obj_id']
        scene_id = bop_data['scene_id']
        im_id = bop_data['im_id']

        # Pre-process the loaded BOP raw data
        data = self.loadData(bop_data, shift=False)
        img, mask, xyz = data['img'], data['mask'], data['xyz']

        mask = mask.copy()

        if self.zephyr_results is not None:
            # Use the mask predicted by zephyr
            zr = self.zephyr_results[(obj_id, scene_id, im_id)]
            mask = zr['pred_mask_visib'].astype(np.float32)[None]

        # From mask get bounding box
        # Will be in (x1, y1, x2, y2) format, where x is rightward and y is downward
        h, w = mask.shape[1:3] # mask in shape (1, h, w)
        mask_pixels = np.stack(mask[0].nonzero(), axis=1)
        y1, x1 = mask_pixels.min(0)
        y2, x2 = mask_pixels.max(0)
        # The last 1 means it is positive and pad an extra dimension
        bbox_gt = np.asarray([[x1, y1, x2, y2, 1]]) 
        # Get the center of the bounding box and convert it to a GT Gaussian heatmap
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        heatmap = heatmapGaussain(
            h*self.heatmap_scale, w*self.heatmap_scale, 
            cx*self.heatmap_scale, cy*self.heatmap_scale, 
            sigma=np.sqrt(1.5)
        )
        heatmap = heatmap[None]

        # Randomly Sample one global template
        gvid = random.choice(self.template_dataset.view_ids)
        gimg, gxyz, gmask = self.template_dataset.getTemplate(obj_id, gvid)

        if self.dataset_mode in ["train"]:
            # return one local template during training time
            # Get the template with closest rotation to the gt rotation
            # gt_rot = [_['obj2cam'][:3, :3] for _ in data['objects'] if _['obj_id'] == int(obj_id)]
            # gt_rot = gt_rot[0][:3, :3]
            gt_rot = bop_data['mat_gt'][:3, :3]
            gt_quat = R.from_matrix(gt_rot).as_quat()
            angular_diff = quatAngularDiffBatch(self.template_dataset.grid_quats, gt_quat[None])
            sorted_view_ids = angular_diff.reshape(-1).argsort()
            lvid = np.random.choice(sorted_view_ids[:self.cfg.train_local_template_sample_from])
            limg, lxyz, lmask = self.template_dataset.getTemplate(obj_id, lvid)
        # elif self.dataset_mode in ['valid'] and not self.cfg.valid_all_templates:
        #     # Return one random local template for validation
        #     lvid = random.choice(self.template_dataset.view_ids)
        #     limg, lxyz, lmask = self.template_dataset.getTemplate(obj_id, lvid)
        elif self.dataset_mode in ['test', 'valid']:
            # Return all templates for this object
            limg, lxyz, lmask = self.template_dataset.getTemplatesAll(obj_id)
            if len(limg) > self.cfg.n_local_test:
                # lvids = random.choices(list(range(len(limg))), k=self.cfg.n_local_test)
                lvids = np.linspace(0, len(limg)-1, self.cfg.n_local_test).round().astype(int)
                # lvids = list(range(len(template_img)))
                limg = limg[lvids]
                lxyz = lxyz[lvids]
                lmask = lmask[lvids]
        else:
            raise Exception("Unknown dataset_mode:", self.dataset_mode)

        out = {
            "img": img, "xyz": xyz, "mask": mask,
            "gimg": gimg, "gxyz": gxyz, "gmask": gmask,
            "limg": limg, "lxyz": lxyz, "lmask": lmask,
            "bbox_gt": bbox_gt, "heatmap": heatmap,
            "obj_id": int(obj_id), 
            "scene_id": scene_id, 
            "im_id": im_id,
        }

        if self.zephyr_results is not None:
            out['zephyr_score'] = zr['score']

        if self.template_dataset.use_provided_template and self.dataset_mode in ['test']:
            out['template_z_values'] = self.template_dataset.template_z_values
        
        return out

