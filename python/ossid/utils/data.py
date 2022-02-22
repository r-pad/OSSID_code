import numpy as np
import cv2
import time
from ossid.utils import normalizeImage, depth2xyz, TorchTimer
from ossid.utils.homographies import sampleWarp

def processData(
        img, mask, depth, cam_K, 
        crop=False, zoom_factor=2.0, crop_shift=False, 
        homographic_warp=False, homographic_down_factor=8,
        homographic_random_kpt = True,
        keep_aspect_ratio = False,
        warp_3d = True, shorter_length = 224,
        ):
    '''
    img: np.ndarray of (H, W, 3) in np.uint8
    mask: np.ndarray of (H, W) in float-like between [0, 1]
    depth: np.ndarray of (H, W) in float-like
    cam_K: np.ndarray of (3, 3) in float-like
    '''
    assert mask.max() <= 1 and mask.min() >= 0
    assert img.dtype == np.uint8
    assert len(img.shape) == 3 and img.shape[2] == 3
    assert len(mask.shape) == 2
    assert len(depth.shape) == 2

    H_ori, W_ori, _ = img.shape

    # depth map to xyz map
    xyz = depth2xyz(depth, cam_K)
    # xyz = xyz - xyz.mean(axis=(0,1), keepdims=True)

    # Crop if told so
    if crop:
        img, mask, xyz = cropMask(img, mask, xyz, zoom_factor, shift=crop_shift)

    # Resize the shorter edge to 224
    if keep_aspect_ratio:
        # print("processData: keeping aspect ratio")
        scale = float(shorter_length) / min(H_ori, W_ori)
        H_resize = round(H_ori * scale // 8) * 8
        W_resize = round(W_ori * scale // 8) * 8
    else:
        H_resize, W_resize = int(shorter_length), int(shorter_length)

    img = cv2.resize(img, (W_resize, H_resize))
    mask = cv2.resize(mask, (W_resize, H_resize))
    xyz = cv2.resize(xyz, (W_resize, H_resize))    
    
    # Camera matrix also needs to be changed
    scale_H = float(H_resize) / H_ori
    scale_W = float(W_resize) / W_ori
    cam_K = cam_K.copy()
    cam_K[1] *= scale_H
    cam_K[0] *= scale_W
    # cam_K[0] *= scale
    # cam_K[1] *= scale

    # do the homographic warp if told so
    out = {}
    if homographic_warp:
        # Do the warping
        kpts, kpts_warp, H, TR, Tt = sampleWarp(
            img, xyz, cam_K, 
            down_factor=homographic_down_factor, 
            random_kpt=homographic_random_kpt,
            warp_3d = warp_3d
        )
        
        out['kpts'], out['kpts_warp'], out['H'], out['TR'], out['Tt'] = \
            kpts, kpts_warp, H.astype(np.float32), TR, Tt

    # Processing for PyTroch
    img = img.transpose(2, 0, 1)
    img = normalizeImage(img).astype(np.float32)
    mask = mask[None].astype(np.float32)
    xyz = xyz.transpose(2, 0, 1).astype(np.float32)

    out['img'] = img
    out['mask'] = mask
    out['xyz'] = xyz

    return out

def cropMask(img_in, mask_in, xyz_in, zoom_factor = 1.0, shift = False):
    h, w, _  = img_in.shape

    img = np.pad(img_in, ((h, h), (w, w), (0,0)), mode='constant', constant_values=img_in.min())
    mask = np.pad(mask_in, ((h, h), (w, w)), mode='constant', constant_values=0)
    xyz = np.pad(xyz_in, ((h, h), (w, w), (0,0)), mode='constant', constant_values=0)
    
    mask_idx = mask.nonzero()

    if mask_idx[0].size == 0:
        return img_in, mask_in, xyz_in
    
    min_x, max_x = mask_idx[0].min(), mask_idx[0].max()
    min_y, max_y = mask_idx[1].min(), mask_idx[1].max()
    mh, mw = max_x - min_x, max_y - min_y
    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2

    r = max(max_x - min_x, max_y - min_y) // 2    
    r = int(zoom_factor * r)

    if shift:
        cx, cy = cx + int(np.random.rand() * r - r/2.0), cy + int(np.random.rand() * r - r/2.0)
        cx = int(np.clip(cx, 1.2*h, 1.8*h))
        cy = int(np.clip(cy, 1.2*w, 1.8*w))
        
    mask = mask[cx-r:cx+r, cy-r:cy+r]
    img = img[cx-r:cx+r, cy-r:cy+r]
    xyz = xyz[cx-r:cx+r, cy-r:cy+r]
    
    return img, mask, xyz

