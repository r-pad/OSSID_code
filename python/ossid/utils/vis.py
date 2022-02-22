import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2

from ossid.utils import to_np

def visBbox(img, pred_bbox=None, pred_score=None, gt_bbox=None, ax = None, topk=3):
    '''
    :param img: img, (H, W, 3), np.uint8
    :param pred_bbox: (K, 4), np.float, in pixels coordinates, (x1, y1, x2, y2)
    :param pred_scores_np: (K, ), np.float, in [0, 1]
    :param gt_bbox: (K', 4), np.float, in pixels coordinates, (x1, y1, x2, y2)
    '''
    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(img)
    ax.axis('off')

    if gt_bbox is not None:
        for box in gt_bbox:
            # print((box[0], [1]), box[2]-box[0], box[3]-box[1])
            gt_rect = patches.Rectangle(
                (box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                linewidth=1, edgecolor='g', facecolor='none'
                )
            ax.add_patch(gt_rect)

    cm = matplotlib.cm.get_cmap('Reds')
    for i in range(min(len(pred_bbox), topk)):
        s = pred_score[i]
        box = pred_bbox[i]
        gt_rect = patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1], 
            linewidth=0.7, edgecolor=cm(pred_score[i]), facecolor='none'
            )
        ax.add_patch(gt_rect)
    
    return ax

def visMask(img, mask, ax, fig, color=[0,0,255], binary=True):
    if img.max() < 2:
        img = (img*255).round().astype(np.uint8)
        
    if binary:
        mask = (mask > 0.5).astype(np.uint8)*255
        mask_edge = cv2.Canny(mask, 1, 1)

        vis = maskimg(img, mask, mask_edge, color=color)
        
        if ax is not None:
            ax.imshow(vis)
            ax.axis('off')
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ax.imshow(gray, cmap='gray')
        ax.axis('off')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        im = ax.imshow(mask, cmap = "rainbow", alpha=0.5)
        fig.colorbar(im, cax=cax, orientation='vertical')
        
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def visKptScores(img, kpt, scores, ax, s=0.1):
    ax.axis('off')
    ax.imshow(img)
    sc = ax.scatter(kpt[:, 0], kpt[:, 1], c = scores, s=s, cmap='rainbow')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(sc, cax=cax)

def plot3dScatter(pts, colors = None, fig=None, ax = None, s=20):
    if ax is None:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=s)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    axisEqual3D(ax)

    return fig, ax

def maskimg(img, mask, edge, color=[0, 0, 255], alpha=0.5):
    '''
    img: cv2 image
    mask: bool or np.where
    color: BGR triplet [_, _, _]. Default: [0, 0, 255] is blue.
    alpha: float [0, 1].

    Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    '''
    out = img.copy()
    img_layer = img.copy()
    img_layer[mask==255] = color
    edge_layer = img.copy()
    edge_layer[edge==255] = color
    out = cv2.addWeighted(edge_layer, 1, out, 0 , 0, out)
    out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)
    return(out)

def plotImages(imgs, titles=None, kpts=None, axes = None):
    if axes is None:
        n = len(imgs)
        fig, axes = plt.subplots(1, n, dpi=300, figsize=(2*n, 2))
    else:
        fig = None
        if type(axes) is np.ndarray:
            axes.reshape(-1)
        else:
            axes = np.array([axes])
        n = min(len(imgs), len(imgs))
        
    for i in range(n):
        ax = axes[i]
        ax.imshow(imgs[i])
        ax.axis('off')
        if titles is not None:
            ax.set_title(titles[i], fontsize=6)

        if kpts is not None:
            for kpt in kpts[i]:
                # kpt = [10, 60]
                c = plt.Circle((kpt[1], kpt[0]), 1, color='r', fc='r')
                # c = plt.Circle((10, 60), 1, color='r', fc='r')
                ax.add_patch(c)
        
    return fig, axes