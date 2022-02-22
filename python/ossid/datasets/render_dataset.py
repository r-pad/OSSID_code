import os
import glob
import numpy as np
import random
import cv2
import h5py
import json
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import binary_erosion

import torch
from torch.utils.data import Dataset, DataLoader

from ossid.datasets.utils import collate_fn, getSampler
from ossid.utils.data import processData
from ossid.utils import depth2xyz, normalizeImage, robustCrop
from ossid.utils.augmentation import augmentDepthMap

def getDataloaders(cfg):
    # Load meta information
    obj2fnames_path = os.path.join(cfg.dataset.dataset_root, "object2files.json")
    obj2fnames = json.load(open(obj2fnames_path, "r"))
    object_ids = list(obj2fnames.keys())

    # Convert the file names to file paths
    obj2paths = {}
    for obj_id in object_ids:
        obj2paths[obj_id] = [os.path.join(cfg.dataset.dataset_root, "%s.hdf5" % fn) for fn in obj2fnames[obj_id]]

    # Split the train, valid and test objects
    train_obj_ids = object_ids[ : len(object_ids)//6*4]
    valid_obj_ids = object_ids[len(object_ids)//6*4 : len(object_ids)//6*5]
    test_obj_ids = object_ids[len(object_ids)//6*5 : ]

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

    # Initialize the DataLoader class
    train_dataset = RenderDataset("train", train_set, cfg.dataset)
    train_loader = DataLoader(
        train_dataset,num_workers=cfg.train.num_workers, collate_fn = collate_fn, pin_memory=True,  
        **(getSampler(train_dataset, cfg.train.batch_size, shuffle=True, ttt_sampling=cfg.dataset.ttt_sampling))
    )
    valseen_dataset = RenderDataset("valid", valseen_set, cfg.dataset)
    valseen_loader = DataLoader(
        valseen_dataset, num_workers=cfg.train.num_workers, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(valseen_dataset, cfg.train.batch_size, shuffle=cfg.train.val_shuffle, ttt_sampling=cfg.dataset.ttt_sampling))
    )
    valunseen_dataset = RenderDataset("valid", valunseen_set, cfg.dataset)
    valunseen_loader = DataLoader(
        valunseen_dataset, num_workers=cfg.train.num_workers, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(valunseen_dataset, cfg.train.batch_size, shuffle=cfg.train.val_shuffle, ttt_sampling=cfg.dataset.ttt_sampling))
    )
    test_dataset = RenderDataset("test", test_set, cfg.dataset)
    test_loader = DataLoader(
        test_dataset, num_workers=cfg.train.num_workers, collate_fn = collate_fn, pin_memory=True, 
        **(getSampler(test_dataset, cfg.train.batch_size, shuffle=cfg.train.val_shuffle, ttt_sampling=cfg.dataset.ttt_sampling))
    )

    print("train imgs:", len(train_loader.dataset), \
          "valseen imgs:", len(valseen_loader.dataset), \
          "valunseen imgs:", len(valunseen_loader.dataset), \
          "test imgs:", len(test_loader.dataset)
        )

    return train_loader, [valunseen_loader, valseen_loader], test_loader

class RenderDataset(Dataset):
    def __init__(self, dataset_mode, obj2paths, cfg) -> None:
        self.dataset_mode = dataset_mode
        self.obj2paths = obj2paths

        self.dataset_root = cfg.dataset_root
        self.k_support = cfg.k_support
        self.crop = cfg.crop
        self.render_support = cfg.render_support
        self.homographic_warp = cfg.homographic_warp
        self.augment_depth = cfg.augment_depth
        self.cfg = cfg

        self.obj_ids = list(obj2paths.keys())

        # Flatten the datapoints
        self.datapoints = []
        for obj_id, paths in self.obj2paths.items():
            self.datapoints += [(obj_id, path) for path in paths]

        if self.render_support:
            self.render_folder = cfg.render_folder
            self.obj2renderpaths = {}
            for obj_id in self.obj_ids:
                self.obj2renderpaths[obj_id] = glob.glob(os.path.join(self.render_folder, "%d" % int(obj_id), "*.hdf5"))
                self.obj2renderpaths[obj_id].sort()

    def __len__(self) -> int:
        return len(self.datapoints)

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
        )

        return out
        
    def __getitem__(self, idx):
        # Load the query images
        obj_id, qpath = self.datapoints[idx]
        qdata = self.loadData(qpath, obj_id, homographic_warp=self.homographic_warp, augment_depth=self.augment_depth)
        qimg, qmask, qxyz = qdata['img'], qdata['mask'], qdata['xyz']

        # Randomly sample some support images. 
        # TODO: think about how to sample support images for validation and testing
        if self.render_support:
            if self.dataset_mode == "train":
                spaths = random.choices(self.obj2renderpaths[obj_id], k=self.k_support)
            else:
                spaths = self.obj2renderpaths[obj_id][:self.k_support]
        else:
            spaths = random.choices([_ for _ in self.obj2paths[obj_id] if _ != qpath], k=self.k_support)

        simgs, smasks, sxyzs = [], [], []
        for spath in spaths:
            sdata = self.loadData(spath, obj_id, homographic_warp=False, augment_depth=self.augment_depth)
            simg, smask, sxyz = sdata['img'], sdata['mask'], sdata['xyz']
            # simg = simg * smask

            simgs.append(simg)
            smasks.append(smask)
            sxyzs.append(sxyz)

        simg = np.concatenate(simgs, axis=0)
        smask = np.concatenate(smasks, axis=0)
        sxyz = np.concatenate(sxyzs, axis=0)

        out = {
            "qimg": qimg,
            "qxyz": qxyz,
            "simg": simg,
            "sxyz": sxyz,
            "smask": smask,
            "qmask": qmask, 
            "obj_id": int(obj_id)
        }

        if self.homographic_warp:
            for k in ['kpts', 'kpts_warp', 'H', 'TR', 'Tt']:
                out['q'+k] = qdata[k]

        return out

def loadHdf5(path):
    with h5py.File(path, 'r') as data:
        campose = json.loads(np.array(data["campose"]).tobytes())
        segmap = np.asarray(data['segmap'])
        colors = np.asarray(data['colors'])
        depth = np.asarray(data['depth'])
        segcolormap = json.loads(np.array(data["segcolormap"]).tobytes())
        object_states = json.loads(np.array(data["object_states"]).tobytes())
        normals = np.asarray(data['normals'])

    # post-process the normal map
    normals = (normals-0.5) * 2 # Put the normal map to the correct scales (from -1 to 1)
    # normals = normals / np.linalg.norm(normals, ord=2, axis=2, keepdims=True)

    # Compute obj2cam transformation matrix for each object in the scene
    cam2world = np.asarray(campose[0]['cam2world_matrix']) # Assume there is only one camera

    # In the Blender camera frame
    #    right = +x, up = +y, backward = +z
    # Now convert it to OpenCV camera frame
    #    right = +x, down = +y, forward = +z
    cam2world[:3, 1] *= -1 # Revert the y axis
    cam2world[:3, 2] *= -2 # Revert the x axis

    world2cam = np.linalg.inv(cam2world)
    objects = []
    for obj in object_states:
        if not obj['name'].startswith("obj"): # Ignore objects that are not of interest
            continue
        obj_translation = np.asarray(obj["location"])
        obj_euler = np.asarray(obj["rotation_euler"])

        # XYZ Euler, XYZ Rotation Order - prone to Gimbal Lock (default).
        obj_rotmat = R.from_euler("XYZ", obj_euler, degrees=False).as_matrix()
        
        obj2world = np.hstack([obj_rotmat, obj_translation.reshape((-1,1))])
        obj2world = np.vstack([obj2world, np.asarray([0,0,0,1])])

        obj2cam = world2cam.dot(obj2world)
        obj_id = int(obj['name'].split("_")[-1].split(".")[0])
        objects.append({
            "obj_id": obj_id,
            "obj2world": obj2world,
            "obj2cam": obj2cam
        })

    data = {
        "campose": campose,
        "segmap": segmap,
        "colors": colors,
        "depth": depth,
        "segcolormap": segcolormap,
        "object_states": object_states,
        "objects": objects,
        "cam2world": cam2world,
        "normals": normals
    }
    
    return data

def processRenderGrid(path, output_size=(128,128)):
    '''
    Input:
        path (str), path to the input hdf5 file
    Output:
        image (ndarray of uint8): cropped render image
        xyz (ndarray of float): cropped XYZ image
        mask (ndarray of float from 0 to 1): cropped mask
    '''
    data = loadHdf5(path)
    cam_K = np.asarray(data['campose'][0]['cam_K'])

    image = data['colors']
    objects = data['objects']
    depth = data['depth']
    segmap = data['segmap']
    segcolormap = data["segcolormap"]

    assert len(segcolormap) == 2
    assert len(objects) == 1
    obj2cam = objects[0]['obj2cam']
    obj_id = objects[0]['obj_id']

    xyz = depth2xyz(depth, cam_K)
    
    # Get mask for this object
    for inst in segcolormap:
        cate_id = int(inst['category_id'])
        if cate_id != obj_id:
            continue
        inst_id = int(inst['idx'])
        cate_mask_channel = int(inst['channel_class'])
        inst_mask_channel = int(inst['channel_instance'])
        mask = np.logical_and(segmap[:, :, cate_mask_channel] == cate_id, segmap[:, :, inst_mask_channel] == inst_id)

    # The mask is slightly larger than the actual region
    mask_temp = binary_erosion(mask)

    # Crop the image and xyz according to the mask
    mask_pixels = np.stack(mask_temp.nonzero(), axis=1)
    if len(mask_pixels) == 0:
        # But sometimes the erosion will make mask all zeros...

        # Crop the image and xyz according to the mask
        mask_pixels = np.stack(mask.nonzero(), axis=1)
    else:
        mask = mask_temp
    
    x1, y1 = mask_pixels.min(0)
    x2, y2 = mask_pixels.max(0)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    r = max(x2-x1, y2-y1, 10) // 2 # Make it at least have some radius
    r = int(r*1.1) # Pad the image a bit
    x1, x2, y1, y2 = cx-r, cx+r, cy-r, cy+r

    mask = mask.astype(np.float)
    # The robustCrop can handle out-of-bound error
    image = robustCrop(image, x1, x2, y1, y2)
    xyz = robustCrop(xyz, x1, x2, y1, y2)
    mask = robustCrop(mask, x1, x2, y1, y2)

    # Only keep the relevant regions
    image = (image * mask[:, :, None]).astype(np.uint8)
    xyz = (xyz * mask[:, :, None])

    # Resize images to (128,128)
    image = cv2.resize(image, output_size)
    xyz = cv2.resize(xyz, output_size)
    mask = cv2.resize(mask, output_size)
    
    results = {
        "image": image,
        "xyz": xyz,
        "mask": mask,
        "obj2cam": obj2cam,
        "obj_id": obj_id,
    }

    return results