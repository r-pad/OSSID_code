import torch
import pytorch_lightning as pl

class MaskRCNN(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()

        import os
        import sys
        sys.path.append("/home/qiaog/src/PyTorch-Simple-MaskRCNN")
        import pytorch_mask_rcnn as pmr

        # from torchvision.models.detection.mask_rcnn import MaskRCNN
        # from torchvision.models.detection.rpn import AnchorGenerator
        # from pytorch_mask_rcnn.model.mask_rcnn import MaskRCNN
        # from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

        self.cfg = config.model
        self.input_resize = (config.dataset.img_h, config.dataset.img_w)
        self.n_classes = config.dataset.n_classes + 1
        
        # anchor_sizes = ((32, ), (64, ), (128, ), (256, ), (512, ))
        # backbone = resnet_fpn_backbone('resnet50', pretrained=config.model.pretrained)
        # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        # self.model = MaskRCNN(backbone=backbone, num_classes=self.n_classes,
        #                         max_size=max(self.input_resize), min_size=min(self.input_resize))
        self.model = pmr.maskrcnn_resnet50(config.model.pretrained, self.n_classes)

        self.metric_keys = ["loss", "loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"]

    def forwardTestTime(self, data):
        image = data['img'][0]
        target_obj_id = data['obj_id'][0].item()
        # image = image.cpu().numpy()
        # image = image[::-1, :, :].copy()
        # image = torch.from_numpy(image).cuda()

        det_result = self(image)

        if len(det_result['boxes']) == 0:
            out = {}
            out['final_bbox'] = [torch.Tensor([[0, 0, image.shape[-1], image.shape[-2]]])]
            out['final_score'] = [torch.Tensor([0])]
            out['segmentation'] = torch.zeros((1, 1, image.shape[-2], image.shape[-1]))
            out['seg_IoU'] = 0
            out['seg_IoU_50'] = 0
            return out

        det_result_mask = det_result['labels'] == target_obj_id
        if det_result_mask.sum() == 0:
            det_result_mask[0] = True

        out = {}
        out['final_bbox'] = [det_result['boxes'][det_result_mask]]
        out['final_score'] = [det_result['scores'][det_result_mask]]
        out['segmentation'] = [det_result['masks'][det_result_mask]][0]

        if out['segmentation'].ndim == 2:
            out['segmentation'] = out['segmentation'].unsqueeze(0)
        if out['segmentation'].ndim == 3:
            out['segmentation'] = out['segmentation'].unsqueeze(0)

        # Compute the metrics
        if "heatmap" in data:
            segmentation = out['segmentation'][0,0]
            # print(data['mask'][0,0].sum())
            with torch.no_grad():
                seg_IoU = pl.metrics.functional.classification.iou(
                    segmentation.detach() > 0.5, data['mask'][0,0].long(), ignore_index=0)
            out['seg_IoU'] = seg_IoU
            out['seg_IoU_50'] = (seg_IoU > 0.5).float()

        return out
        
    def forward(self, image, target=None):
        if type(image) is list:
            image = image[0]
        if type(target) is list:
            target = target[0]

        out = self.model(image, target)
        if self.training:
            out['loss'] = sum(loss for loss in out.values())
        return out

    def training_step(self, batch, batch_idx):
        images, targets = batch
        out = self(images[0], targets[0])
        
        loss = sum(loss for loss in out.values())

        out['loss'] = loss

        for k in self.metric_keys:
            if k in out:
                if type(out[k]) is float:
                    v = torch.Tensor([out[k]]).squeeze()
                else:
                    v = out[k].detach().cpu().squeeze()
                self.log("train_"+k, v, on_step=True, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.cfg.learning_rate,
                                    momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.cfg.lr_epoch_decay,
                                                    gamma=0.1)
        return [optimizer], [scheduler]
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, targets = batch
        out = self(images[0])
        out = [out]

        # Compute the Segmentation IoU
        seg_IoUs = []
        for i in range(len(images)):
            target = targets[i]
            result = out[i]
            for gt_id, obj_id in enumerate(target['labels']):
                detected = False
                for detect_id, detect_obj_id in enumerate(result['labels']):
                    if detect_obj_id.item() == obj_id.item():
                        detected = True
                        break

                if detected:
                    detect_mask = result['masks'][detect_id]
                    gt_mask = target['masks'][gt_id]

                    seg_IoU = pl.metrics.functional.classification.iou(detect_mask.detach() > 0.5, gt_mask.long(), ignore_index=0)
                    seg_IoU = seg_IoU.item()
                else:
                    seg_IoU = 0.0

                seg_IoUs.append(seg_IoU)

        seg_IoUs = torch.as_tensor(seg_IoUs)
        mean_seg_iou = seg_IoUs.float().mean()
        seg_iou_50 = (seg_IoUs > 0.5).float().mean()

        self.log("val_mean_seg_iou", mean_seg_iou, logger=True, on_step=False, on_epoch=True)
        self.log("val_seg_iou_50", seg_iou_50, logger=True, on_step=False, on_epoch=True)

        # return out
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

if __name__ == "__main__":
    from omegaconf import DictConfig, OmegaConf
    from ossid.models import getModel
    from ossid.datasets import getDataloaders

    cfg = OmegaConf.load('/home/qiaog/src/feature_graph/python/feature_graph/conf/config.yaml')
    cfg.model = OmegaConf.load('/home/qiaog/src/feature_graph/python/feature_graph/conf/model/maskrcnn.yaml')
    cfg.dataset = OmegaConf.load('/home/qiaog/src/feature_graph/python/feature_graph/conf/dataset/detect.yaml')
    cfg.train.batch_size = 1
    cfg.train.num_workers = 0
    cfg.dataset.load_zephyr_result=True

    train_loader, valid_loader, test_loader = getDataloaders(cfg)
    model = getModel(cfg)

    ckpt = torch.load("/home/qiaog/src/feature_graph/python/feature_graph/lightning_logs/maskrcnn_detect_zephyrlmo_semi/ckpts_v0/last.ckpt")

    model.load_state_dict(ckpt['state_dict'])

    trainer = pl.Trainer(gpus = "0")
    # trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)