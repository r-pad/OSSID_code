import os
from ossid.datasets.utils import getSampler
import numpy as np
from numpy.lib.type_check import imag
import torch
import time
import torchvision.transforms as transforms

from scipy.spatial.transform import Rotation as R

def expandBox(x1, y1, x2, y2, img_h, img_w, expand_ratio):
    cx, cy = (x1+x2) / 2, (y1+y2) / 2
    w, h = x2-x1, y2-y1
    x1, x2 = max(0, cx - w / 2 * expand_ratio), min(img_w-1, cx + w / 2 * expand_ratio)
    y1, y2 = max(0, cy - h / 2 * expand_ratio), min(img_h-1, cy + h / 2 * expand_ratio)
    return x1, y1, x2, y2

def quatAngularDiffBatch(Q1, Q2):
    '''
    Q1 is of shape (M, 4) and Q2 is of shape (N, 4)
    return a matrix of (M, N) containing angular difference between them
    '''
    M, _ = Q1.shape
    N, _ = Q2.shape
    Q1 = torch.from_numpy(Q1)
    Q2 = torch.from_numpy(Q2)
    Q1 = Q1.reshape((M, 4, 1))
    Q2 = Q2.T.reshape((1, 4, N))
    product = torch.abs((Q1*Q2).sum(axis=1))
    angle_diff = 2*torch.acos(torch.min(product, torch.ones_like(product) * 1-1e-7))
    return to_np(angle_diff)

def normalizeImageRange(img):
    '''
    img: torch.Tensor of size (B, 3, H, W)
    '''
    img = (img - img.new_tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1) ) \
        / img.new_tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
    return img

# def denormalizeImageRange(img):
#     '''
#     image: ndarray of size (3, H, W)
#     '''
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     mean = np.asarray(mean).reshape((3, 1, 1))
#     std = np.asarray(std).reshape((3, 1, 1))
#     img = img * std + mean
#     return img
    
def normalizeImage(img):
    '''
    Arguments:
        img: image of shape (3, H, W), range (0, 255)
    '''
    img = img.astype(np.float32)
    img = img / 255.0 # ToTensor
    # img = (img - np.asarray((0.485, 0.456, 0.406)).reshape((3, 1, 1))) \
    #     / np.asarray((0.229, 0.224, 0.225)).reshape((3, 1, 1)) # Normalize
    return img

def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)

    Input: 
        image: torch.Tensor of size (3, H, W), normalized by the mean and variance from ImageNet
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image

def perturbTrans(mat, n_perturb = 500):
    rot_mag = np.random.normal(0, 0.2, n_perturb)
    rot_axis = np.random.normal(0, 1.0, (n_perturb, 3))
    rot_axis = rot_axis / np.linalg.norm(rot_axis, ord=2, axis=1, keepdims=True)

    rotvec = rot_axis * rot_mag[:, None]

    rot_perturb = R.from_rotvec(rotvec)
    rot_perturb = rot_perturb.as_matrix()
    trans_perturb = np.random.normal(0, 0.01, (n_perturb, 3))

    mat_new = mat.copy()[None]
    mat_new = np.repeat(mat_new, n_perturb, axis=0)

    mat_new[:, :3, :3] = np.einsum("ijk,ikl->ijl", rot_perturb, mat_new[:, :3, :3])
    mat_new[:, :3, 3] += trans_perturb
    return mat_new

def randRotMat(Z_max=90, X_max=30, Y_max=30):
    Z_angle = np.random.uniform(-Z_max, Z_max, None)
    X_angle = np.random.uniform(-X_max, X_max, None)
    Y_angle = np.random.uniform(-Y_max, Y_max, None)
    rot_mat = R.from_euler('ZXY', [Z_angle, X_angle, Y_angle], degrees=True).as_matrix()
    return rot_mat

def estimateRigidBodyTransform(P, Q):
    '''
    Compute the rigid body transformation R and t given two set of
    N corresponding points in 3D.
    Inputs:
        P - a (3, N) matrix containing the before-transform points
        Q - a (3, N) matrix containing the after-transform points
    Outputs:
        R, t
    '''
    d, N = P.shape
    p_cen = P.mean(axis = 1).reshape((d, 1))
    q_cen = Q.mean(axis = 1).reshape((d, 1))
    X = P - p_cen
    Y = Q - q_cen
    S = X.dot(Y.T)
    u, sigma, vh = np.linalg.svd(S)
    U = u
    V = vh.T
    middle = np.eye(d)
    middle[-1, -1] = np.linalg.det(V.dot(U.T))
    R = V.dot(middle).dot(U.T)
    t = q_cen - R.dot(p_cen)
    return R, t

def meta2K(meta_data):
    if type(meta_data['camera_fx']) is torch.Tensor:
        cam_K = np.asarray([
            [meta_data['camera_fx'].item(), 0, meta_data['camera_cx'].item()],
            [0, meta_data['camera_fy'].item(), meta_data['camera_cy'].item()],
            [0, 0, 1]
        ])
    else:
        cam_K = np.asarray([
            [meta_data['camera_fx'], 0, meta_data['camera_cx']],
            [0, meta_data['camera_fy'], meta_data['camera_cy']],
            [0, 0, 1]
        ])

    return cam_K

def K2meta(cam_K):
    meta_data = {
        "camera_fx": cam_K[0,0],
        "camera_fy": cam_K[1,1],
        "camera_cx": cam_K[0,2],
        "camera_cy": cam_K[1,2],
        "camera_scale": 1.0
    }
    return meta_data

def dict_to(dictionary, device):
    for k,v in dictionary.items():
        if(type(v) is torch.Tensor):
            dictionary[k]=v.to(device)

def torch_norm_fast(tensor, axis):
    return torch.sqrt((tensor**2).sum(axis))

def to_np(x):
    if type(x) is np.ndarray or type(x) is float or type(x) is int:
        return x

    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    else:
        return x.detach().data.cpu().numpy()

def torch2Img(img, normalized = False):
    disp_img = to_np(img)
    if len(disp_img.shape) == 4:
        disp_img = disp_img[0]
    disp_img = disp_img.transpose((1,2,0))
    if(normalized):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        disp_img = disp_img * std + mean
    return disp_img

class TorchTimer:
    def __init__(self, heading = None, agg_list = None, verbose = True):
        self.verbose = verbose
        if not self.verbose:
            return
        if(agg_list is None and heading is None):
            heading = ""
        self.agg_list = agg_list
        self.heading = heading
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if not self.verbose:
            return self
        self.start.record()
        self.start_cpu = time.time()
        return self

    def __exit__(self, *args):
        if not self.verbose:
            return
        self.end.record()
        torch.cuda.synchronize()
        self.interval_cpu = time.time() - self.start_cpu
        self.interval = self.start.elapsed_time(self.end)/1000.0
        if(self.agg_list is not None):
            if(self.heading is not None):
                self.agg_list.append((self.heading, self.interval, self.interval_cpu))
            else:
                self.agg_list.append((self.interval, self.interval_cpu))
        if (self.heading is not None and self.verbose):
            print('{} GPU:{}, CPU:{}'.format(self.heading, self.interval, self.interval_cpu))

class Timer:
    def __init__(self, heading = "", agg_list = None, verbose = True):
        self.verbose = verbose
        if not self.verbose:
            return
        self.heading = heading

    def __enter__(self):
        if not self.verbose:
            return self
        self.start = time.time()
        return self

    def __exit__(self, *args):
        if not self.verbose:
            return
        self.end = time.time()
        self.interval = self.end - self.start
        print(self.heading, self.interval)


def depth2xyz(depth, cam_K):
    h, w = depth.shape
    ymap, xmap = np.meshgrid(np.arange(w), np.arange(h))

    # Here rightward is the positive x direction
    # And downward is the postive y direction
    x = ymap
    y = xmap
    z = depth

    x = (x - cam_K[0,2]) * z / cam_K[0,0]
    y = (y - cam_K[1,2]) * z / cam_K[1,1]

    xyz = np.stack([x, y, z], axis=2)
    return xyz

def kpts2cloud(kpts, depth, cam_K):
    raise Exception("This function seems wrong (about x and y)")
    x = kpts[:, 0]
    y = kpts[:, 1]
    z = depth[x, y]

    x = (x - cam_K[0,2]) * z / cam_K[0,0]
    y = (y - cam_K[1,2]) * z / cam_K[1,1]

    P_w = np.vstack((x, y, z)).T
    return P_w

def projCloud(pts, cam_K):
    '''
    Project a point cloud in 3D into 2D image plane
    Note in the camera coordinate, rightward is the positive x, and downward is the positive y

    pts: (n, 3) points in 3D, relative to the camera coordinate frame
    cam_K: matrix of camera intrinsics, with the entry [2,2] being 1
    '''
    # x, y are in the camera coordinate
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    # px and py are in the image coordinate (down x, right y)
    py = (cam_K[0,0] * x / z) + cam_K[0,2]
    px = (cam_K[1,1] * y / z) + cam_K[1,2]

    P = np.vstack((px, py)).T
    return P

def torch_norm_fast(tensor, axis):
    return torch.sqrt((tensor**2).sum(axis))

def dict_to(data, device):
    for k,v in data.items():
        if(type(v) is torch.Tensor):
            data[k]=v.to(device)

def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")

def cosSim(mdesc0, mdesc1, axis=1):
    assert mdesc0.dim() == 3
    assert mdesc1.dim() == 3
    assert axis in [1, 2]
    if axis == 1:
        dot = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
    elif axis == 2:
        dot = torch.einsum('bnd,bmd->bnm', mdesc0, mdesc1)
    denom = torch_norm_fast(mdesc0, axis).unsqueeze(2) * torch_norm_fast(mdesc1, axis).unsqueeze(1)
    scores = dot / denom
    return scores

# Q1 is of shape (M, 4) and Q2 is of shape (N, 4)
# return a matrix of (M, N) containing angular difference between them
def quatAngularDiffBatch(Q1, Q2):
    M, _ = Q1.shape
    N, _ = Q2.shape
    Q1 = Q1.reshape((M, 4, 1))
    Q2 = Q2.T.reshape((1, 4, N))
    product = np.absolute((Q1*Q2).sum(axis=1))
    angle_diff = 2*np.arccos(np.minimum(product, np.ones_like(product) * 1-1e-7))
    return angle_diff

def makeDir(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

def robustCrop(image, x1, x2, y1, y2):
    assert x2 > x1
    assert y2 > y1
    from_h, from_w = image.shape[:2]
    to_h, to_w = x2 - x1, y2 - y1
    crop = np.zeros((to_h, to_w, *(image.shape[2:])), dtype=image.dtype)
    
    from_x1, from_y1 = max(0, x1), max(0, y1)
    from_x2, from_y2 = min(from_h, x2), min(from_w, y2)
    to_x1, to_y1 = max(0, -x1), max(0, -y1)
    to_x2, to_y2 = min(to_h, from_h-x1), min(to_w, from_w-y1)
    crop[to_x1:to_x2, to_y1:to_y2] = image[from_x1:from_x2, from_y1:from_y2]
    return crop

def heatmapGaussain(img_h, img_w, cx, cy, sigma, normalize=False):
    img_h, img_w = int(round(img_h)), int(round(img_w))
    # Initializing value of x-axis and y-axis
    # in the range -1 to 1
    x, y = np.meshgrid(np.arange(img_w), np.arange(img_h))
    dst = np.sqrt((x-cx)**2 + (y-cy)**2)

    # Calculating Gaussian array
    gauss = np.exp(-(dst**2 / ( 2.0 * sigma**2 ) ) )

    if normalize:
        gauss = gauss / gauss.sum()

    return gauss