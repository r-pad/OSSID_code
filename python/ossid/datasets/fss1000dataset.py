import imageio
import os
import glob
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader

def getDataloaders(cfg):
    classes_all = glob.glob(os.path.join(cfg.dataset.dataset_root, "*"))
    classes_all = [c.split("/")[-1] for c in classes_all]
    classes_test = open(cfg.dataset.test_split_file, 'r').readlines()
    classes_test = [c.rstrip() for c in classes_test]

    classes_nontest = list(set(classes_all) - set(classes_test))
    num_train = 520
    num_nontest = len(classes_nontest)

    np.random.seed(0) # Set seed for splitting datasets
    classes_train = np.asarray(classes_nontest)[np.random.permutation(num_nontest)[:num_train]]
    classes_train = list(classes_train)
    classes_valid = list(set(classes_nontest) - set(classes_train))

    print("train:", len(classes_train), "valid:", len(classes_valid), "test:", len(classes_test))

    train_set = FSS1000Dataset("train", classes_train, cfg)
    valid_set = FSS1000Dataset("valid", classes_valid, cfg)
    test_set = FSS1000Dataset("test", classes_test, cfg)

    train_loader = DataLoader(
        train_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=cfg.train.batch_size, shuffle=cfg.train.val_shuffle, num_workers=cfg.train.num_workers,
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.train.batch_size, shuffle=cfg.train.val_shuffle, num_workers=cfg.train.num_workers,
    )

    return train_loader, valid_loader, test_loader

class FSS1000Dataset(Dataset):
    def __init__(self, dataset_mode, class_list, cfg):
        self.dataset_mode = dataset_mode
        self.class_list = class_list

        self.dataset_root = cfg.dataset.dataset_root
        self.k_support = cfg.dataset.k_support

    def __len__(self):
        return len(self.class_list) * 5

    def processData(self, img, mask):
        img = np.asarray(img).astype(float)
        mask = np.asarray(mask).astype(float)

        img = cv2.resize(img, (224, 224))
        mask = cv2.resize(mask, (224, 224))

        mask = (mask[:, :, 0][None] / 255)

        img = img.transpose(2, 0, 1)
        img = img / 255.0 # ToTensor
        img = (img - np.asarray((0.485, 0.456, 0.406)).reshape((3, 1, 1))) \
              / np.asarray((0.229, 0.224, 0.225)).reshape((3, 1, 1)) # Normalize

        return img, mask

    def __getitem__(self, idx):
        class_idx = idx // 5
        img_idx = idx % 5

        class_name = self.class_list[class_idx]
        class_folder = os.path.join(self.dataset_root, class_name)

        if self.dataset_mode == "train":
            permut = np.random.permutation(10) + 1
            snums = permut[:self.k_support]
            qnum = permut[self.k_support]
            # snum, qnum = np.random.permutation(10)[:2] + 1
        else:
            snums = list(range(1, 1+self.k_support))
            qnum = img_idx + 6
            # snum, qnum = img_idx + 1, img_idx + 6

        simgs = []
        smasks = []
        for snum in snums:
            simg = imageio.imread(os.path.join(class_folder, "%d.jpg" % snum))
            smask = imageio.imread(os.path.join(class_folder, "%d.png" % snum))
            simg, smask = self.processData(simg, smask)
            simgs.append(simg)
            smasks.append(smask)

        simg = np.concatenate(simgs, axis=0)
        smask = np.concatenate(smasks, axis=0)

        qimg = imageio.imread(os.path.join(class_folder, "%d.jpg" % qnum))
        qmask = imageio.imread(os.path.join(class_folder, "%d.png" % qnum))

        qimg, qmask = self.processData(qimg, qmask)

        # print(simg.shape, simg.dtype, simg.max(), simg.min())
        # print(smask.shape, smask.dtype, smask.max(), smask.min())
        # print(qimg.shape, qimg.dtype, qimg.max(), qimg.min())
        # print(qmask.shape, qmask.dtype, qmask.max(), qmask.min())

        data = {
            "simg": simg,
            "qimg": qimg,
            "smask": smask,
            "qmask": qmask
        }

        return data
