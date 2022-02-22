import torch
# from pytorch_lightning.metrics import Metric

class MatchPrecision():
    def __init__(self, name = None):
        self.name = name

    def __call__(self, scores, gt_scores, match_thresh = 0.2):
        '''
        scores: (n_batch, n_obs_kpt, n_model_kpt)
        gt_scores: (n_batch, n_obs_kpt, n_model_kpt)
        The dustbins should be already removed from the input
        '''
        assert scores.shape == gt_scores.shape

        # Return 0 if no matching are detected
        if scores.sum() == 0:
            return torch.tensor(0, device=scores.device, dtype=scores.dtype)

        # scores = scores > match_thresh
        return (scores * gt_scores).sum() / scores.sum()

class MatchRecall():
    def __init__(self, name = None):
        self.name = name

    def __call__(self, scores, gt_scores, match_thresh = 0.2):
        '''
        scores: (n_batch, n_obs_kpt, n_model_kpt)
        gt_scores: (n_batch, n_obs_kpt, n_model_kpt)
        The dustbins should be already removed from the input
        '''
        assert scores.shape == gt_scores.shape

        # If there is no match in the groud truth
        if gt_scores.sum() == 0:
            if scores.sum() == 0:
                return torch.tensor(1, device=scores.device, dtype=scores.dtype)
            else:
                return torch.tensor(0, device=scores.device, dtype=scores.dtype)

        # scores = scores > match_thresh
        return (scores * gt_scores).sum() / gt_scores.sum()

class ObsSegIoU():
    def __init__(self, name = None):
        self.name = name

    def __call__(self, scores, gt_scores, match_thresh = 0.1):
        '''
        scores: (n_batch, n_obs_kpt, n_model_kpt)
        gt_scores: (n_batch, n_obs_kpt, n_model_kpt)
        The dustbins should be already removed from the input
        '''
        assert scores.shape == gt_scores.shape
        match_thresh = torch.tensor([match_thresh], device=scores.device, dtype=scores.dtype)

        gt_seg = gt_scores.max(-1)[0] >= match_thresh
        seg = scores.max(-1)[0] >= match_thresh
        inter = torch.logical_and(gt_seg, seg).sum().float()
        union = torch.logical_or(gt_seg, seg).sum().float()
        if union == 0:
            iou = torch.tensor(0, device=scores.device, dtype=scores.dtype)
        else:
            iou = inter / union

        return iou
