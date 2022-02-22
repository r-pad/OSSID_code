'''
Source: 
    https://github.com/ethz-asl/multipoint/blob/1dec30339bee456a4aa2fee843a51b919635818d/multipoint/utils/homographies.py
    https://github.com/rpautrat/SuperPoint/blob/master/superpoint/models/homographies.py
'''

import numpy as np
import cv2
import torch
from functools import reduce
import kornia
import random
from ossid.utils import depth2xyz, projCloud, randRotMat, estimateRigidBodyTransform

def warpKornia(src, H):
    '''
    Input:
        src: Tensor of size (N, C, H, W)
        H: Tensor of size (N, 3, 3)
    '''
    _, _, h, w = src.size
    warp = kornia.warp_perspective(src, H, dsize=(h, w))
    return warp

def warpTorch3D(img, xyz, HomoT, TR, Tt):
    '''
    img: Tensor of size (B, 3, H, W)
    xyz: Tensor of size (B, 3, H, W)
    HomoT: homographic transformation matrix of size (B, 3, 3)
    TR: 3D rotation of size (B, 3, 3)
    Tt: 3D translation of size (B, 3, 1)
    '''
    B, _, H, W = img.shape
    
    img_warp = kornia.warp_perspective(img, HomoT, dsize=(H, W))
    
    if xyz is None:
        xyz_warp = None
    else:
        depth_mask = (xyz[:, 2:3, :, :] != 0).float().expand(B, 3, H, W) # valid depth (B, 1, H, Warp)

        xyz_warp = kornia.warp_perspective(xyz, HomoT, dsize=(H, W))
        depth_mask_warp = kornia.warp_perspective(depth_mask, HomoT, dsize=(H, W))
        
        xyz_warp = torch.einsum("bpq,bqij->bpij", TR, xyz_warp) + Tt.reshape((B, 3, 1, 1))
        xyz_warp[depth_mask_warp == 0] = 0 # reset invalid depth to 0

    return img_warp, xyz_warp

def sampleWarp(img, xyz, cam_K, n_kpts=128, down_factor=8, random_kpt = True, warp_3d = True):
    h, w = img.shape[:2]
    
    if random_kpt:
        # print("Random keypoints")
        kpts = np.stack([np.random.randint(h, size=n_kpts//2), np.random.randint(w, size=n_kpts//2)], axis=1)
    else:
        # print("SIFT detecting")
        # Detect the keypoints from the original image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        kpts_cv = sift.detect(gray,None)
        # downsample the keypoints according to the detection scores
        if len(kpts_cv) > n_kpts:
            kpts_responses = np.asarray([_.response for _ in kpts_cv])
            kpts_cv = np.random.choice(kpts_cv, size = n_kpts, replace=False, p=kpts_responses/kpts_responses.sum())
            kpts = cv2.KeyPoint_convert(kpts_cv)[:, ::-1].astype(int) # Convert to numpy convention in terms of x, y
            kpts = filter_points(kpts, (h, w))
        elif len(kpts_cv) > 0:
            kpts = cv2.KeyPoint_convert(kpts_cv)[:, ::-1].astype(int) # Convert to numpy convention in terms of x, y
            kpts = filter_points(kpts, (h, w))
        else: # No detected keypoints from SIFT
            kpts = np.stack([np.random.randint(h, size=n_kpts), np.random.randint(w, size=n_kpts)], axis=1)
    
    # Sample a homographic transformation and warp the RGB/XYZ/mask/keypoints
    # H = sample_homography((h, w))
    if warp_3d:
        H, TR, Tt = sampleTrans3D(xyz, cam_K)
    else:
        H = sample_homography((h, w))
        TR, Tt = np.eye(3), np.zeros((3, 1))

    TR, Tt = TR.astype(np.float32), Tt.astype(np.float32)
    
    kpts_warp = warp_keypoints(kpts, H)

    idx3 = filter_points_return_indices(kpts_warp, (h, w))

    # Scale the coordinates according to the down_factor, and filter
    assert kpts_warp.shape == kpts.shape
    kpts = kpts // down_factor
    kpts_warp = kpts_warp // down_factor
    _, idx1 = np.unique(kpts, return_index=True, axis=0)
    _, idx2 = np.unique(kpts_warp, return_index=True, axis=0)

    idx = np.intersect1d(idx1, idx2, assume_unique=True)
    idx = np.intersect1d(idx, idx3, assume_unique=True)

    kpts = kpts[idx]
    kpts_warp = kpts_warp[idx]

    return kpts, kpts_warp, H, TR, Tt

def sampleTrans3D(xyz, cam_K):
    '''
    Sample a 3D rigid transformation as well as the corresponding transformation in 2D
    NP-coord: down x, right y
    CV-coord: right x, down y
    cam-coord: right x, down y, depth z

    input:
        img: np.ndarray of the image, (H, W, 3)
        xyz: np.ndarray of the xyz map, (H, W, 3)
        cam_K: np.ndarray of the camera intrinsics, (3, 3)
    '''
    # Get the range of x, y translation
    x_span = xyz[:, :, 0].max(None) - xyz[:, :, 0].min(None)
    y_span = xyz[:, :, 1].max(None) - xyz[:, :, 1].min(None)

    # Get the anchor points
    mean = xyz.sum((0, 1)) / (xyz[:, :, -1] != 0).sum()
    mean_d = mean + np.asarray([0.41, 0, 0.1])
    mean_r = mean + np.asarray([0, 0.42, 0.2])
    mean_dr = mean + np.asarray([0.43, 0.44, -0.15])
    pts1 = np.stack([mean, mean_r, mean_d, mean_dr])
    pts1_proj = projCloud(pts1, cam_K)

    # When the estimateRigidBodyTransform fails, re-sample the rotation
    while True:
        pts2 = pts1.copy()

        # Apply the rotation
        rot_mat = randRotMat(X_max=40, Y_max=40)
        rot_center = mean.reshape(-1, 1)
        pts2 = ((rot_mat.dot(pts2.T - rot_center)) + rot_center).T # rotation is repective to the center of the point cloud

        # Apply the translation
        trans_vec = np.asarray([
            (random.random()-0.5) * y_span * 0.2,
            (random.random()-0.5) * x_span * 0.2,
            random.random() * mean[2],
        ])
        pts2 = pts2 + trans_vec.reshape((1,-1))

        try:
            TR, Tt = estimateRigidBodyTransform(pts1.T, pts2.T)
        except np.linalg.LinAlgError:
            print("sampleTrans3D: Bad rotation, re-try")
            continue

        break
        
    pts2_proj = projCloud(pts2, cam_K)

    # Estimate the 2D homography and 3D rigid body transformation
    homography = cv2.getPerspectiveTransform(pts1_proj.astype(np.float32)[:, ::-1], pts2_proj.astype(np.float32)[:, ::-1])


    return homography, TR, Tt

'''
Options in multipoint:
    perspective=True, scaling=True, rotation=True, translation=True,
    n_scales=10, n_angles=25, scaling_amplitude=0.2, perspective_amplitude_x=0.1,
    perspective_amplitude_y=0.1, patch_ratio=0.8, max_angle=np.pi/2,
    allow_artifacts=True, translation_overflow=0.1

Options in superpoint:
    perspective=True, scaling=True, rotation=True, translation=True,
    n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
    perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=np.pi/2,
    allow_artifacts=False, translation_overflow=0.
'''
def sample_homography(image_shape, perspective=True, scaling=True, rotation=True, translation=True,
                      n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
                      perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=np.pi/2,
                      allow_artifacts=True, translation_overflow=0.1):
    """
    Sample a random valid homography.
    Arguments:
        image_shape: The shape of the image
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.
    Returns:
        A numpy array containing the homographic transformation matrix
    """

    def transform_perspective(points):
        t_min, t_max = -points.min(axis=0), 1.0-points.max(axis=0)
        t_max[1] = min(abs(t_min[1]), abs(t_max[1]))
        t_min[1] = -t_max[1]
        if not allow_artifacts:
            perspective_amplitude_min = np.maximum(np.array([-perspective_amplitude_x,-perspective_amplitude_y]), t_min)
            perspective_amplitude_max = np.minimum(np.array([perspective_amplitude_x,perspective_amplitude_y]), t_max)
        else:
            perspective_amplitude_min = np.array([-perspective_amplitude_x,-perspective_amplitude_y])
            perspective_amplitude_max = np.array([perspective_amplitude_x,perspective_amplitude_y])

        perspective_displacement = np.random.uniform(perspective_amplitude_min[1], perspective_amplitude_max[1])
        h_displacement_left = np.random.uniform(perspective_amplitude_min[0], perspective_amplitude_max[0])
        h_displacement_right = np.random.uniform(perspective_amplitude_min[0], perspective_amplitude_max[0])

        tmp = points.copy()
        points += np.array([[h_displacement_left,   perspective_displacement],
                          [h_displacement_left,  -perspective_displacement],
                          [h_displacement_right,  perspective_displacement],
                          [h_displacement_right, -perspective_displacement]])

        return points

    def transform_scale(points):
        scales = np.random.uniform(-scaling_amplitude, scaling_amplitude, n_scales) + 1.0
        center = points.mean(axis=0)
        scaled = np.expand_dims(points - center, 0) * np.expand_dims(np.expand_dims(scales, 1), 1) + center

        if allow_artifacts:
            valid = np.arange(n_scales)  # all scales are valid except scale=1
        else:
            valid = []
            for i in range(n_scales):
                if scaled[i,...].max() < 1.0 and scaled[i,...].min() >= 0.0:
                    valid.append(i)

        if valid is not None:
            idx = np.random.choice(valid)
            points = scaled[idx]
        else:
            print('sample_homography: No valid scale found')

        return points

    def transform_translation(points):
        t_min, t_max = -points.min(axis=0), 1.0-points.max(axis=0)
        if allow_artifacts:
            t_min -= translation_overflow
            t_max += translation_overflow
        points += np.array([np.random.uniform(t_min[0], t_max[0]),
                          np.random.uniform(t_min[1], t_max[1])])

        return points

    def transform_rotation(points):
        angles = np.random.uniform(-max_angle, max_angle, n_angles)
        angles = np.append(angles, 0)  # in case no rotation is valid
        center = points.mean(axis=0)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul(np.tile(np.expand_dims(points - center, axis=0), [n_angles+1, 1, 1]), rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles)  # all angles are valid, except angle=0
        else:
            valid = []
            for i in range(len(angles)):
                if rotated[i,...].max() < 1.0 and rotated[i,...].min() >= 0.0:
                    valid.append(i)

        idx = np.random.choice(valid)
        points = rotated[idx]

        return points

    # Corners of the input image
    pts1 = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])

    # Corners of the output patch
    margin = (1 - patch_ratio) * 0.5
    pts2 = margin + patch_ratio * pts1

    # Random perspective and affine perturbations
    functions = []
    if perspective:
        functions.append(transform_perspective)

    # Random scaling
    if scaling:
        functions.append(transform_scale)

    # Random translation
    if translation:
        functions.append(transform_translation)

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        functions.append(transform_rotation)

    indices = np.arange(len(functions))
    np.random.shuffle(indices)

    for i in range(len(functions)):
            idx = indices[i]
            pts2 = functions[idx](pts2)

    # Rescale to actual size
    shape = image_shape[::-1]  # different convention [y, x]
    pts1 *= shape
    pts2 *= shape

    homography = cv2.getPerspectiveTransform(pts1.astype(np.float32), pts2.astype(np.float32))
    return homography

def warp_keypoints(keypoints, homography, return_type=np.int):
    """
    Warp the keypoints based on the specified homographic transformation matrix
    Arguments:
        keypoints: Array containing the keypoints, shape: [N,2]
        homography: 3x3 transformation matrix
    Returns: Array containing the warped keypoints, shape: [N,2]
    """
    if len(keypoints) > 0:
        warped_points = cv2.perspectiveTransform(np.array([keypoints[:,::-1]], dtype=np.float64), homography)
        return warped_points[0,:,::-1].astype(return_type)
    else:
        # no keypoints available so return the empty array
        return keypoints

def warp_points_pytorch(points, homography):
    # get the points to the homogeneous format
    warped_points = torch.cat([points.flip(-1), torch.ones([points.shape[0], points.shape[1], 1], dtype=torch.float32, device=points.device)], -1)

    # apply homography
    warped_points = torch.bmm(homography, warped_points.permute([0,2,1])).permute([0,2,1])
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]

    return warped_points.flip(-1)

def filter_points(points, shape):
    """
    Filter points which would be outside the image frame
    Arguments:
        points: Array containing the keypoints, shape: [N,2]
        shape: Image shape
    Returns: Array containing the filtered keypoints, shape: [M,2]
    """
    points = points[points[:,0] >= 0]
    points = points[points[:,1] >= 0]
    points = points[points[:,0] < shape[0]]
    points = points[points[:,1] < shape[1]]

    return points

def filter_points_return_indices(points, shape):
    """
    Filter points which would be outside the image frame
    Arguments:
        points: Array containing the keypoints, shape: [N,2]
        shape: Image shape
    Returns: Array containing the filtered keypoints, shape: [M,2]
    """
    idx1 = points[:,0] >= 0
    idx2 = points[:,1] >= 0
    idx3 = points[:,0] < shape[0]
    idx4 = points[:,1] < shape[1]
    idx = reduce(np.logical_and, [idx1, idx2, idx3, idx4])

    return idx.nonzero()[0]

def compute_valid_mask(image_shape, homography, erosion_radius=0, mask_border = False):
    """
    Compute a boolean mask of the valid pixels resulting from an homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.
 
    Arguments:
        input_shape: Array of rank 2 representing the image shape, i.e. `[H, W]`.
        homography: Array of shape (3, 3)
        erosion_radius: radius of the margin to be discarded.
        mask_border: Boolean indicating if the border is used to erode the valid region 
 
    Returns: Array of shape (H, W).
    """
    mask = cv2.warpPerspective(np.ones(image_shape), homography, image_shape[::-1], flags=cv2.INTER_NEAREST)

    if erosion_radius > 0:
        if mask_border:
            tmp = np.zeros((image_shape[0]+2, image_shape[1]+2))
            tmp[1:-1,1:-1] = mask
            mask = tmp
        kernel = np.ones((erosion_radius * 2 + 1,erosion_radius * 2 + 1),np.float32)
        mask = cv2.erode(mask,kernel,iterations = 1)

        if mask_border:
            mask = mask[1:-1,1:-1]

    return mask