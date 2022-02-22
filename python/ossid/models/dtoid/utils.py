import torch
from torchvision.ops import nms


def nonMaxSup(classifications, transformed_anchors, nms_iou_thresh=0.5):
    '''
    Perform the non-maximum suppression on the direct output of the network 

    :param classifications: (B, N, C), probability of classification, where 0 is background
    :param transformed_anchors: (B, N, 4)
    :param nms_iou_thresh: number, IoU threshold used for NMS
    '''
    B, N, C = classifications.shape # batch size, number of anchor boxes, number of classes
    
    # Each Tensor in the following lists corresponding to one image in the batch
    final_score = [classifications.new_tensor([]) for b in range(B)]
    final_class = [classifications.new_tensor([]) for b in range(B)]
    final_bbox = [classifications.new_tensor([]) for b in range(B)]
    
    for b in range(B):
        this_score = []
        this_class = []
        this_bbox = []
        for c in range(1, C): # (ignore the background class)
            scores = classifications[b, :, c] # (N, )
            scores_over_thresh = (scores > 0.05) # (n, )
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue
                
            scores = scores[scores_over_thresh] # (n, )
            anchorBoxes = transformed_anchors[b, scores_over_thresh] # (n, 4)
            anchors_nms_idx = nms(anchorBoxes, scores, nms_iou_thresh) # (n')

            this_score.append(scores[anchors_nms_idx])    
            this_class.append(classifications.new_tensor([c] * anchors_nms_idx.shape[0]))
            this_bbox.append(anchorBoxes[anchors_nms_idx])
    
        if len(this_score) == 0:
            # No positive output, continue
            continue

        final_score[b] = torch.cat(this_score, 0)
        final_class[b] = torch.cat(this_class, 0)
        final_bbox[b] = torch.cat(this_bbox, 0)

    return final_score, final_class, final_bbox
