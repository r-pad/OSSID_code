import numpy as np
import random
import cv2

def augmentDepthMap(depth, normals):
    H, W = depth.shape

    # Randomly corrupt the depth edges
    threshold = random.uniform(0.2, 0.5)
    # projection = (normals * np.asarray([0,0,1]).reshape((1,1,3))).sum(-1)
    # mask = np.ones_like(depth)
    # mask[projection < threshold] = 0
    projection = normals[:, :, 2]
    mask = projection >= threshold
    
    # Randomly corrupt rectangular patches
    for i in range(random.randint(0, 6)):
        x1 = np.random.randint(H)
        y1 = np.random.randint(W)
        x2 = min(H-1, x1 + np.random.randint(H//16, H//4))
        y2 = min(W-1, y1 + np.random.randint(W//16, W//4))
        mask[x1:x2, y1:y2] = 0

    depth = depth * mask

    return depth, mask