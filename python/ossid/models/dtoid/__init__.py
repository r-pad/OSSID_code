import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops import nms
import pytorch_lightning as pl

from .network import Network, BBoxTransform, ClipBoxes

from ossid.utils import normalizeImageRange, to_np, TorchTimer
from ossid.utils.vis import plotImages, visBbox
from ossid.models.dtoid.loss import DetectionLoss
from ossid.models.dtoid.utils import nonMaxSup
from ossid.utils.detection_metrics import DetectionMetric

TIMER_VERBOSE = False

def tensor_list_to(list_of_tensor, device):
    new_list = [_.to(device) for _ in list_of_tensor]
    return new_list

class DtoidNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.cfg = config.model
        self.img_size = (self.cfg.img_h, self.cfg.img_w)
        self.heatmap_size = (self.cfg.heatmap_h, self.cfg.heatmap_w)

        # Initialize the network
        self.model = Network(img_size=self.img_size, heatmap_size=self.heatmap_size)

        if self.cfg.use_pretrained_dtoid:
            print("Loading pretrained DTOID weights from:", self.cfg.pretrained_dtoid_path)
            ckpt = torch.load(self.cfg.pretrained_dtoid_path)
            self.load_pretrained_state_dict(ckpt['state_dict'])

        # utils
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.det_map = DetectionMetric(['fg'])

        self.det_loss_func = DetectionLoss()
        self.center_loss_func = torch.nn.L1Loss()
        self.seg_loss_func = torch.nn.BCELoss()

        self.metric_keys = ['loss', 'loss_seg', 'loss_center', 'loss_cls', 'loss_reg', "seg_IoU", 'seg_IoU_50'] #, "mAP"]
        self.metric_keys_all_template = ['seg_IoU', 'seg_IoU_50'] # metrics used at test time

        # Cache for template features, only used during test time
        self.template_feature_cache = {}

    def load_pretrained_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def clearCache(self):
        del self.template_feature_cache
        self.template_feature_cache = {}

    def forwardTestTime(self, input):
        image = input['img']
        # Forward one image and one object at a time 
        assert len(image) == 1

        # ImageNet RGB normalization
        image = normalizeImageRange(image)

        obj_id = input['obj_id'][0].item() # The object ID to be detected

        # Get template features, either from cache or compute online
        if obj_id not in self.template_feature_cache:
            # Forward templates of this object to get features
            # For the local templates, the batch dimension is 160
            # Remove the extra dimension added by pytorch dataloader
            template = input['limg'][0]
            template_mask = input['lmask'][0]

            # ImageNet RGB normalization
            template = normalizeImageRange(template)

            # concatenate template image with its mask
            template = torch.cat([template, template_mask], dim = 1)

            # Compute the features for the global template
            # which corresponds to the first local template
            with torch.no_grad():
                template_feature_global = self.model.compute_template_global(template[0:1])
            template_global_list = [template_feature_global]

            # Split template input into chunks and compute template feature for each of them
            chunk_size = 120
            chunks = torch.split(template, chunk_size, dim=0)
            feature_chunks = []
            for chunk in chunks:
                with torch.no_grad():
                    feature_chunk = self.model.compute_template_local(chunk)
                feature_chunks.append(feature_chunk)
            
            template_list = feature_chunks

            # self.template_feature_cache[obj_id] = (
            #     tensor_list_to(template_list, image.device), 
            #     tensor_list_to(template_global_list, image.device)
            # )

            self.template_feature_cache[obj_id] = (
                tensor_list_to(template_list, 'cpu'), 
                tensor_list_to(template_global_list, 'cpu')
            )

        else:
            template_list, template_global_list = self.template_feature_cache[obj_id]
            template_list = tensor_list_to(template_list, image.device)
            template_global_list = tensor_list_to(template_global_list, image.device)
        
        top_k_num = 500
        with torch.no_grad():
            with TorchTimer("forward_all_templates", verbose=TIMER_VERBOSE):
                top_k_scores, top_k_bboxes, top_k_template_ids, seg_pred, heatmap_pred = self.model.forward_all_templates(
                    image, template_list, template_global_list, topk=top_k_num
                )
        
            # Filter predictions according to estimated z values
            if 'template_z_values'  in input and self.cfg.filter_z:
                pred_bbox_np = top_k_bboxes.cpu().numpy()
                pred_template_ids = top_k_template_ids[:, 0].long().cpu().numpy()
                template_z_values = input['template_z_values'].cpu().numpy()[0, pred_template_ids]

                pred_w_np = pred_bbox_np[:, 2] - pred_bbox_np[:, 0]
                pred_h_np = pred_bbox_np[:, 3] - pred_bbox_np[:, 1]
                pred_max_dim_np = np.stack([pred_w_np, pred_h_np]).transpose().max(axis=1)
                pred_z = (124 / pred_max_dim_np) * -template_z_values

                # Filter based on predicted Z values
                pred_z_conds = (pred_z > 0.4) & (pred_z < 2)
                pred_z_conds_ids = np.where(pred_z_conds)[0]
                if len(pred_z_conds_ids) == 0:
                    pred_z_conds_ids = [0]

                top_k_scores = top_k_scores[pred_z_conds_ids]
                top_k_bboxes = top_k_bboxes[pred_z_conds_ids]
                top_k_template_ids = top_k_template_ids[pred_z_conds_ids]
                seg_pred = seg_pred[pred_z_conds_ids]
                heatmap_pred = heatmap_pred[pred_z_conds_ids]
                pred_z = pred_z[pred_z_conds_ids]

            top_k_template_ids = top_k_template_ids[:, 0]
            seg_pred = torch.sigmoid(seg_pred)


        out = {
            "pred_bbox": top_k_bboxes, # (K, 4)
            "pred_scores": top_k_scores, # (K, )
            "pred_template_ids": top_k_template_ids, # (k, )
            "segmentation": seg_pred.unsqueeze(1), # (K, 1, img_h, img_w)
            "heat_map": heatmap_pred.unsqueeze(1), # (K, 1, heatmap_h, heatmap_w)
            "final_bbox": [top_k_bboxes], # list of (k, 4)
            "final_score": [top_k_scores], # list of (k, )
        }

        # Compute the metrics
        if "heatmap" in input:
            segmentation = seg_pred[0]
            with torch.no_grad():
                seg_IoU = pl.metrics.functional.classification.iou(
                    segmentation.detach() > 0.5, input['mask'][0,0].long(), ignore_index=0)
            out['seg_IoU'] = seg_IoU
            out['seg_IoU_50'] = (seg_IoU > 0.5).float()

        return out

    def forward(self, input):
        image = input['img']
        template = input['limg']
        template_mask = input['lmask']
        global_template = input['gimg']
        global_template_mask = input['gmask']

        # Assert the input RGB images are normalized to [0, 1]
        for img in image, template, global_template:
            assert img.max() <= 1
            assert img.min() >= 0
            
        # Normalization using ImageNet mean and variance
        image = normalizeImageRange(image)
        template = normalizeImageRange(template)
        global_template = normalizeImageRange(global_template)

        # Forward pass of the underlying DTOID model
        classifications, regressions, anchors, heat_map, segmentation = \
            self.model(image, template, template_mask, global_template, global_template_mask)
        
        segmentation = torch.sigmoid(segmentation)

        out = {}
        out["classifications"] = classifications
        out["regressions"] = regressions
        out["anchors"] = anchors
        out["heat_map"] = heat_map
        out["segmentation"] = segmentation

        # Anchor-based detection post procesing
        # Add the bbox regressions to the anchors and clip according to image dimensions
        transformed_anchors = self.regressBoxes(anchors, regressions)
        transformed_anchors = self.clipBoxes(transformed_anchors, image)

        out["transformed_anchors"] = transformed_anchors

        # If ground truth is available
        if "heatmap" in input:
            # Compute the loss
            loss_cls, loss_reg = self.det_loss_func(classifications, regressions, anchors, input['bbox_gt'].to(0))
            loss_center = self.center_loss_func(input['heatmap'], heat_map)
            loss_seg = self.seg_loss_func(segmentation, input['mask'])
            # loss_seg = 0

            loss = self.cfg.lam_seg * loss_seg + \
                   self.cfg.lam_center * loss_center + \
                   self.cfg.lam_cls * loss_cls + \
                   self.cfg.lam_reg * loss_reg

            out['loss_seg'] = self.cfg.lam_seg * loss_seg
            out['loss_center'] = self.cfg.lam_center * loss_center
            out['loss_cls'] = self.cfg.lam_cls * loss_cls
            out['loss_reg'] = self.cfg.lam_reg * loss_reg
            out['loss'] = loss

            with torch.no_grad():
                seg_IoU = pl.metrics.functional.classification.iou(segmentation.detach() > 0.5, input['mask'].long(), ignore_index=0, reduction = 'none')
                mean_seg_iou = seg_IoU.mean()
                seg_iou_50 = (seg_IoU.mean(dim = list(range(1, seg_IoU.dim()))) > 0.5).float().mean()

                out['seg_IoU'] = mean_seg_iou
                out['seg_IoU_50'] = seg_iou_50
            
                final_score, final_class, final_bbox = nonMaxSup(out['classifications'].detach(), out['transformed_anchors'].detach())

                out['final_score'] = final_score # list of (k, )
                out['final_class'] = final_class # list of (k, )
                out['final_bbox'] = final_bbox # list of (k, 4)
            
                # APs, mAP = self.det_map.calculate_mAP(
                #     final_bbox, final_class, final_score, 
                #     list(input['bbox_gt'][:, :, :4].detach()), list(input['bbox_gt'][:, :, 4].detach())
                # )
                # out['mAP'] = mAP

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr = self.cfg.learning_rate, 
            weight_decay = self.cfg.weight_decay,
            amsgrad = True
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

        return [optimizer], [scheduler]

    def visInOut(self, d, out, idx = 0):
        final_bbox = out['final_bbox']
        final_score = out['final_score']
        heat_map = out['heat_map']
        segmentation = out['segmentation']

        img_np = to_np(d['img'][idx]).transpose(1,2,0)*255
        img_np = img_np.round().astype(np.uint8)

        gimg_np = to_np(d['gimg'][idx]).transpose(1,2,0)*255
        gimg_np = gimg_np.round().astype(np.uint8)

        limg_np = to_np(d['limg'][idx])
        if limg_np.ndim > 3:
            limg_np = limg_np[0] # Sample the first local template in all templates
        limg_np = limg_np.transpose(1,2,0)*255
        limg_np = limg_np.round().astype(np.uint8)

        gt_bbox = to_np(d['bbox_gt'])[idx, :, :4]

        pred_bbox = to_np(final_bbox[idx])
        pred_score = to_np(final_score[idx])

        fig, axes = plt.subplots(2, 4, dpi=200, figsize = (8, 4))
        axes = axes.reshape(-1)

        plotImages(
            [gimg_np, limg_np], 
            ["G temp", "L temp"], 
            axes = axes[1:3]
        )

        axes[3].remove()

        plotImages(
            [to_np(d['heatmap'][0,0]), to_np(heat_map[0,0]), to_np(d['mask'][0,0]), to_np(segmentation[0,0])], 
            ['GT heatmap', 'pred heatmap', 'GT mask', 'pred mask'], 
            axes = axes[4:8]
        )

        axes[0].set_title("Detected boxes", fontsize=6)
        visBbox(img_np, pred_bbox=pred_bbox, pred_score=pred_score, gt_bbox=gt_bbox, ax=axes[0], topk=5)

        return fig, axes

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = out['loss']

        for k in self.metric_keys:
            if k in out:
                if type(out[k]) is float:
                    v = torch.Tensor([out[k]]).squeeze()
                else:
                    v = out[k].detach().cpu().squeeze()
                self.log("train_"+k, v, on_step=True, on_epoch=True, logger=True)

        if batch_idx % 500 == 0:
            fig, axes = self.visInOut(batch, out)
            self.logger.experiment.log({"train_vis": fig})
            plt.close(fig)

        return loss

    def on_validation_epoch_start(self) -> None:
        # Clean up the cache
        self.clearCache()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.cfg.valid_all_templates:
            out = self.forwardTestTime(batch)
            metric_keys = self.metric_keys_all_template
        else:
            out = self(batch)
            metric_keys = self.metric_keys

        result = {}
        for k in metric_keys:
            if k in out:
                if type(out[k]) is float:
                    v = torch.Tensor([out[k]]).squeeze()
                else:
                    v = out[k].detach().cpu().squeeze()
                result[k] = v

        if batch_idx % 250 == 0:
            fig, axes = self.visInOut(batch, out)
            if dataloader_idx == 0:
                self.logger.experiment.log({"valunseen_vis_%06d" % batch_idx: fig})
            else:
                self.logger.experiment.log({"valseen_vis_%06d" % batch_idx: fig})
            plt.close(fig)

        return result

    def validation_epoch_end(self, outputs):
        if self.cfg.valid_all_templates:
            metric_keys = self.metric_keys_all_template
        else:
            metric_keys = self.metric_keys
        
        def mean(outs, key):
            return torch.mean(torch.stack([_[key] for _ in outs]))

        if type(outputs[0]) is list:
            for k in metric_keys:
                if k in outputs[0][0]:
                    self.log("valunseen_"+k, mean(outputs[0], k), logger=True)
                    self.log("valseen_"+k, mean(outputs[1], k), logger=True)
        else:
            # log the same value for seen and unseen objects
            for k in metric_keys:
                if k in outputs[0]:
                    self.log("valunseen_"+k, mean(outputs, k), logger=True)
                    self.log("valseen_"+k, mean(outputs, k), logger=True)

    def on_test_epoch_start(self) -> None:
        # Clean up the cache
        self.clearCache()

    def test_step(self, batch, batch_idx):
        out = self.forwardTestTime(batch)
        metric_keys = self.metric_keys_all_template

        result = {}
        for k in metric_keys:
            if k in out:
                result[k] = out[k].detach().cpu()

        for k in ['obj_id', 'im_id', 'scene_id']:
            if k in batch:
                result[k] = batch[k].item()

        return result

    def test_epoch_end(self, outputs):
        metric_keys = self.metric_keys_all_template
        for k in metric_keys:
            v = torch.tensor([o[k] for o in outputs]).mean().item()
            # print("mean %s:" % k, v)
            self.log("test_%s" % k, v)