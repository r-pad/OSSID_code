import sys
import argparse
import os
import cv2
import yaml
from PIL import Image
from importlib.machinery import SourceFileLoader
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas
import numpy

__filedir__ = os.path.dirname(os.path.realpath(__file__))

# network_module = SourceFileLoader(".", os.path.join(__filedir__, "network.py")).load_module()
import feature_graph.models.dtoid.network as network_module


class DTOIDWrapper(nn.Module):
    def __init__(self, backend="cuda", no_filter_z=False):
        super(DTOIDWrapper, self).__init__()

        # Initialize the network
        model = network_module.Network()
        model.eval()
        # model_path = os.path.join(__filedir__, "model.pth.tar")
        model_path = "/home/qiaog/src/DTOID/model.pth.tar"
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["state_dict"])

        if backend == "cuda":
            model = model.cuda()

        self.model = model
        self.backend = backend
        self.no_filter_z = no_filter_z
        self.preprocess = network_module.PREPROCESS
        # self.model_directory = os.path.join(__filedir__, "templates")
        self.model_directory = "/home/qiaog/src/DTOID/templates"
        self.template_cache = {}

    def clearCache(self):
        del self.template_cache
        self.template_cache = {}

    def getTemplates(self, linemod_model):
        '''
        linemod_model: str of the linemod object ID ("01", "02", ...)
        '''
        if linemod_model in self.template_cache:
            return
        
        assert type(linemod_model) is str

        model_name = "hinterstoisser_" + linemod_model
        template_dir = os.path.join(self.model_directory, model_name)
        output_file = "{}.yml".format(model_name)

        #load text file
        pose_file = os.path.join(template_dir, "poses.txt")
        pose_file_np = pandas.read_csv(pose_file, sep=" ", header=None).values
        pose_z_values = pose_file_np[:, 11]

        # Template
        global_template_list = []
        template_paths = [x for x in os.listdir(template_dir) if len(x) == 12 and "_a.png" in x]
        template_paths.sort()
        preprocessed_templates = []

        # features for all templates (240)
        template_list = []
        template_global_list = []
        template_ratios_list = []

        batch_size = 10
        temp_batch_local = []
        temp_batch_global = []
        temp_batch_ratios = []
        iteration = 0

        for t in tqdm(template_paths):
            # open template and template mask
            template_im = cv2.imread(os.path.join(template_dir, t))[:, :, ::-1]
            template = Image.fromarray(template_im)

            template_mask = cv2.imread(os.path.join(template_dir, t.replace("_a", "_m")))[:, :, 0]
            template_mask = Image.fromarray(template_mask)

            # preprocess and concatenate
            template = self.preprocess[1](template)
            template_mask = self.preprocess[2](template_mask)
            template = torch.cat([template, template_mask], dim=0)

            if self.backend == "cuda":
                template = template.cuda()

            template_feature = self.model.compute_template_local(template.unsqueeze(0))

            # Create mini-batches of templates
            if iteration == 0:
                temp_batch_local = template_feature

                template_feature_global = self.model.compute_template_global(template.unsqueeze(0))
                template_global_list.append(template_feature_global)

            elif iteration % (batch_size) == 0:
                template_list.append(temp_batch_local)
                temp_batch_local = template_feature

            elif iteration == (len(template_paths) - 1):
                temp_batch_local = torch.cat([temp_batch_local, template_feature], dim=0)
                template_list.append(temp_batch_local)

            else:
                temp_batch_local= torch.cat([temp_batch_local, template_feature], dim=0)

            iteration += 1

        self.template_cache[linemod_model] = (template_list, template_global_list, pose_z_values)

    def forward(self, img_numpy, obj_id):
        template_list, template_global_list, pose_z_values = self.template_cache[obj_id]

        img_h, img_w, img_c = img_numpy.shape
        img = Image.fromarray(img_numpy)

        img = self.preprocess[0](img)

        network_h = img.size(1)
        network_w = img.size(2)
        if self.backend == "cuda":
            img = img.cuda()

        top_k_num = 500
        top_k_scores, top_k_bboxes, top_k_template_ids, seg_pred = self.model.forward_all_templates(
            img.unsqueeze(0), template_list, template_global_list, topk=top_k_num)


        pred_seg_np = seg_pred.cpu().numpy()
        pred_scores_np = top_k_scores.cpu().numpy()
        pred_bbox_np = top_k_bboxes.cpu().numpy()
        pred_template_ids = top_k_template_ids[:, 0].long().cpu().numpy()
        template_z_values = pose_z_values[pred_template_ids]

        if not self.no_filter_z:
                    
            pred_w_np = pred_bbox_np[:, 2] - pred_bbox_np[:, 0]
            pred_h_np = pred_bbox_np[:, 3] - pred_bbox_np[:, 1]
            pred_max_dim_np = np.stack([pred_w_np, pred_h_np]).transpose().max(axis=1)
            pred_z = (124 / pred_max_dim_np) * -template_z_values

            # Filter based on predicted Z values
            pred_z_conds = (pred_z > 0.4) & (pred_z < 2)
            pred_z_conds_ids = numpy.where(pred_z_conds)[0]

            pred_scores_np = pred_scores_np[pred_z_conds_ids]
            pred_bbox_np = pred_bbox_np[pred_z_conds_ids]
            pred_template_ids = pred_template_ids[pred_z_conds_ids]
            pred_z = pred_z[pred_z_conds_ids]


        # Keep top 1 (eval)
        pred_scores_np = pred_scores_np[:1]
        pred_bbox_np = pred_bbox_np[:1]
        pred_template_ids = pred_template_ids[:1]
        pred_z = pred_z[:1]
        pred_seg_np = pred_seg_np[:1]

        output = {
            "pred_bbox_np": pred_bbox_np,
            "pred_scores_np": pred_scores_np,
            "pred_seg_np": pred_seg_np,
            "pred_template_ids": pred_template_ids,
            "network_w": network_w,
            "network_h": network_h,
            "img_h": img_h,
            "img_w": img_w,
        }
        
        return output


