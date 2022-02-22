'''
reference: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
'''

import pickle
from networkx.algorithms.operators.unary import reverse
import numpy as np
import os
import random
import torch
import argparse
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, DataLoader, dataloader

from ossid.utils import meta2K, quatAngularDiffBatch, heatmapGaussain
from ossid.utils.constants import BOP_OBJECT_ID_OFFSETS
from ossid.utils.data import processData
from ossid.datasets.utils import collate_fn, getSampler, loadProcessZephyrResults, sortTargetByImage
from ossid.datasets.template_dataset import TemplateDataset

from zephyr.utils.bop_dataset import BopDataset

from . import transforms as T
from .utils import collate_fn


def getDataloaders(cfg):
    print("here")
    # Initialize the Bop Dataset for testing
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
        cfg.dataset.zephyr_result_path = "/home/qiaog/datasets/bop/zephyr_results/test_lmo_boptest_zephyr_result.pkl"
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
        cfg.dataset.zephyr_result_path = "/home/qiaog/datasets/bop/zephyr_results/test_ycbv_boptest_zephyr_result_unseen.pkl"
        cfg.dataset.n_classes = 21
    else:
        raise Exception("Unknown cfg.dataset.test_dataset_name:", cfg.dataset.test_dataset_name)

    test_bop_dataset = BopDataset(test_args)
    # unseen_objects = test_bop_dataset.obj_ids
    unseen_objects = list(range(1, cfg.dataset.n_classes + 1))

    # Initialize the Bop Dataset for training
    if cfg.dataset.train_dataset_name is None or cfg.dataset.train_dataset_name == cfg.dataset.test_dataset_name:
        print("Train BOP dataset: the %s test set" % test_args.dataset_name)
        train_bop_dataset = deepcopy(test_bop_dataset)

        # Initialize dataste for finetuning using zephyr results
        if cfg.dataset.load_zephyr_result:
            zephyr_results_train, zephyr_results_valid = loadProcessZephyrResults(cfg)
        else:
            zephyr_results_train, zephyr_results_valid = None, None

        seen_objects = unseen_objects
    else:
        raise Exception("Unknown cfg.dataset.train_dataset_name: %s" % cfg.dataset.train_dataset_name)

    train_transform = get_transform(True)
    test_transform = get_transform(False)
    train_dataset = DetectDataset("train", seen_objects, train_bop_dataset, train_transform, zephyr_results_train)
    train_loader = DataLoader(train_dataset, 
        batch_size=cfg.train.batch_size, shuffle=False, 
        num_workers=cfg.train.num_workers, collate_fn=collate_fn)

    # Initialize the validation Dataset and DataLoader
    valid_dataset = DetectDataset('valid', seen_objects, train_bop_dataset, test_transform, zephyr_results_valid)
    valid_loader = DataLoader(valid_dataset, 
        batch_size=cfg.train.batch_size, shuffle=False, 
        num_workers=cfg.train.num_workers, collate_fn=collate_fn)

    # Initialize the testing Dataset and DataLoader
    test_dataset = DetectDataset('test', unseen_objects, test_bop_dataset, test_transform)
    test_loader = DataLoader(test_dataset, 
        batch_size=cfg.train.batch_size, shuffle=False, 
        num_workers=cfg.train.num_workers, collate_fn=collate_fn)

    print(
        "train_dataset:", len(train_dataset), "train_loader:", len(train_loader), 
        "\nval_dataset:", len(valid_dataset), "val_loader:", len(valid_loader),
        "\ntest_dataset:", len(test_dataset), "test_loader:", len(test_loader),
    )

    # Vaildation loader will be the same as the test loader here
    return train_loader, test_loader, test_loader

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class DetectDataset(Dataset):
    def __init__(
            self, 
            dataset_mode: str, 
            obj_ids, 
            bop_dataset: BopDataset, 
            transforms, 
            zephyr_results=None
        ) -> None:
        super().__init__() # For Dataset class, this is meaningless
        self.dataset_mode = dataset_mode
        self.obj_ids = obj_ids

        self.bop_dataset = deepcopy(bop_dataset)
        self.transforms = transforms
        self.dataset_name = self.bop_dataset.dataset_name

        self.targets_detect_all = sortTargetByImage(self.bop_dataset.targets)

        # filter the BopDataset by zephyr results if passed in
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

        # Group the object presences by (scene_id and im_id)
        self.targets_detect = sortTargetByImage(self.bop_dataset.targets)
        self.targets_detect = [(k[0], k[1], v) for k, v in self.targets_detect.items()]

    def clearTargets(self):
        '''
        remove all datapoints
        '''
        self.bop_dataset.targets = []
        self.targets_detect = []
    
    def sortTargets(self, reverse=False):
        self.bop_dataset.targets.sort(
            reverse = reverse,
            key = lambda x: (x['scene_id'], x['im_id'], x['obj_id'])
        )
        print("Sorted the testing targets", self.bop_dataset.targets[:3])

        # Group the object presences by (scene_id and im_id)
        self.targets_detect = sortTargetByImage(self.bop_dataset.targets)
        self.targets_detect = [(k[0], k[1], v) for k, v in self.targets_detect.items()]
    
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

        # Group the object presences by (scene_id and im_id)
        self.targets_detect = sortTargetByImage(self.bop_dataset.targets)
        self.targets_detect = [(k[0], k[1], v) for k, v in self.targets_detect.items()]

    def updateZephyrMask(self, obj_id, scene_id, im_id, mask, score):
        '''
        Update the pseudo ground-truth in the zephyr results for better finetuning
        '''
        self.zephyr_results[(obj_id, scene_id, im_id)]['pred_mask_visib'] = mask
        self.zephyr_results[(obj_id, scene_id, im_id)]['score'] = score

    def __len__(self):
        return len (self.targets_detect)

    def __getitem__(self, idx):
        target = self.targets_detect[idx]
        scene_id, im_id, obj_ids = target

        obj_confidence = torch.ones(len(self.obj_ids)+1)
        if self.zephyr_results is not None:
            for obj_id in self.targets_detect_all[(scene_id, im_id)]:
                if obj_id not in obj_ids:
                    obj_confidence[obj_id] = 0

        # get the base data
        bop_data = self.bop_dataset.getDataByIds(obj_ids[0], scene_id, im_id)
        img = bop_data['img'] # np.ndarray of (H, W, 3) in np.uint8

        # get the bbox, label and mask for each object
        boxes = []
        labels = []
        masks = []
        num_objs = len(obj_ids)
        for obj_id in obj_ids:
            mask_data = self.bop_dataset.getMaskByIds(obj_id, scene_id, im_id)
            mask = mask_data['mask_gt_visib']

            if self.zephyr_results is not None:
                # Use the mask predicted by zephyr
                zr = self.zephyr_results[(obj_id, scene_id, im_id)]
                mask = zr['pred_mask_visib'].astype(np.float32)

            h, w = mask.shape 
            mask_pixels = np.stack(mask.nonzero(), axis=1)
            y1, x1 = mask_pixels.min(0)
            y2, x2 = mask_pixels.max(0)

            # Convert the mask to be between 0 and 1
            if mask.max() > 10:
                mask = mask / 255

            boxes.append([x1, y1, x2, y2])
            labels.append(obj_id)
            masks.append(mask)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)

        image_id = torch.tensor([scene_id*100000 + im_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target['cls_conf'] = obj_confidence

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
        