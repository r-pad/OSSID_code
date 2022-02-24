import open3d as o3d
import os, sys
import argparse

import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import torch
import pickle
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader

from ossid.models.dtoid import DtoidNet
from ossid.models.maskrcnn import MaskRCNN
from ossid.datasets import getDataloaders
from ossid.datasets.utils import collate_fn
from ossid.utils import expandBox, dict_to, to_np, move_to
from ossid.utils.bop_utils import saveResultsBop
from ossid.utils.zephyr_utils import networkInference
from ossid.config import OSSID_CKPT_ROOT, OSSID_DATA_ROOT, BOP_RESULTS_FOLDER, OSSID_RESULT_ROOT, BOP_DATASETS_ROOT, OSSID_DET_ROOT
from ossid.utils.detection import saveLmoYcbvGT, evalFinetuneResults

from zephyr.datasets.score_dataset import ScoreDataset
from zephyr.models.pointnet2 import PointNet2SSG
from zephyr.options import getOptions, checkArgs
from zephyr.utils import depth2cloud, meta2K, K2meta, projectPointsUv
from zephyr.utils.metrics import add, adi
from zephyr.utils.bop_dataset import BopDataset, BopDatasetArgs
from zephyr.utils.halcon_wrapper import PPFModel
from zephyr.utils.renderer import Renderer, blend
from zephyr.utils.icp import icpRefinement
from zephyr.constants import OBJECT_DIAMETERES
from zephyr.data_util import hypoShiftYcbv2BopBatch, modelPointsShiftYcbv2Bop, modelShiftBopYcbv

from zephyr.full_pipeline.model_featurization import FeatureModel
from zephyr.full_pipeline.scene_featurization import featurizeScene

from bop_toolkit_lib.visibility import estimate_visib_mask_gt
from bop_toolkit_lib.misc import ensure_dir, depth_im_to_dist_im_fast

import faulthandler
faulthandler.enable()

def makeFolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def getFeaturizedModels(dataset):
    from zephyr.full_pipeline.options import getOptions
    parser = getOptions()
    args = parser.parse_args([])

    args.bop_root = dataset.bop_root
    args.dataset_name = dataset.dataset_name
    args.grid_dir_name = "grid"
    args.sampled_model_dir_name = "model_pc"
    args.grid_indices_path = os.path.join(args.bop_root, args.dataset_name, args.grid_dir_name, "verts_grid_0.npy")
    
    dataset.dataset_camera["fx"] = dataset.dataset_camera['K'][0,0]
    dataset.dataset_camera["fy"] = dataset.dataset_camera['K'][1,1]
    dataset.dataset_camera["cx"] = dataset.dataset_camera['K'][0,2]
    dataset.dataset_camera["cy"] = dataset.dataset_camera['K'][1,2]
    
    featured_objects = {}
    for obj_id in dataset.obj_ids:
        is_sym = obj_id in dataset.sym_obj_ids
        obj = FeatureModel(dataset.dataset_root, is_sym, args, create_index=True)
        obj.construct(obj_id, dataset.getObjPath(obj_id), dataset.dataset_camera)
        featured_objects[obj_id] = obj
        
    return featured_objects

def main(main_args):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    DATASET_NAME = main_args.dataset_name
    DTOID_CONFIDENT_THRESHOLD = 0.5
    ZEPHYR_CONFIDENT_THRESHOLD = 20
    SAVE_ROOT = OSSID_RESULT_ROOT
    assert not (main_args.ignore_dtoid_mask and main_args.always_dtoid_mask)

    makeFolder(SAVE_ROOT)
    makeFolder(BOP_RESULTS_FOLDER)

    next_finetune_number = main_args.finetune_interval

    '''Initialize the trained DTOID model'''
    # Use the DTOID network
    if main_args.dtoid_weights_path is not None:
        ckpt_v = int(main_args.dtoid_weights_path.split("/")[-2].split("_")[1][1:])
        ckpt_path = Path(main_args.dtoid_weights_path)
        conf_path = ckpt_path.parent.parent / ("config_v%d.yaml" % ckpt_v)
    elif DATASET_NAME == 'lmo':
        conf_path = os.path.join(OSSID_CKPT_ROOT, "dtoid_conf_lmo.yaml")
        if main_args.use_offline_model:
            ckpt_path = os.path.join(OSSID_CKPT_ROOT, "dtoid_transductive_lmo.ckpt")
        else:
            ckpt_path = os.path.join(OSSID_CKPT_ROOT, "dtoid_pretrained.ckpt")
    elif DATASET_NAME == 'ycbv':
        conf_path = os.path.join(OSSID_CKPT_ROOT, "dtoid_conf_ycbv.yaml")
        if main_args.use_offline_model:
            ckpt_path = os.path.join(OSSID_CKPT_ROOT, "dtoid_transductive_ycbv.ckpt")
        else:
            ckpt_path = os.path.join(OSSID_CKPT_ROOT, "dtoid_pretrained.ckpt")

    ossid_args = OmegaConf.load(conf_path)

    # Override arguments by use-provided directories
    ossid_args.dataset.bop_root = BOP_DATASETS_ROOT
    ossid_args.model.pretrained_dtoid_path = os.path.join(OSSID_CKPT_ROOT, "dtoid_pretrained_original.pth.tar")
    if DATASET_NAME == 'ycbv':
        ossid_args.dataset.grid_root = os.path.join(OSSID_DATA_ROOT, "templates_YCBV_BOP")
        ossid_args.dataset.zephyr_result_path = os.path.join(OSSID_DATA_ROOT, "test_ycbv_boptest_zephyr_result_unseen.pkl")
    elif DATASET_NAME == 'lmo':
        ossid_args.dataset.grid_root = os.path.join(OSSID_DATA_ROOT, "templates_LMO_DTOID")
        ossid_args.dataset.zephyr_result_path = os.path.join(OSSID_DATA_ROOT, "lmo_boptest_zephyr_result.pkl")

    # Use the DTOID provided by original authors (https://github.com/jpmerc/DTOID)
    # This model was trained also on YCB-V objects, and thus can only be used to evaluate on LM-O. 
    ossid_args.model.use_pretrained_dtoid = main_args.use_pretrained_dtoid

    ossid_args.dataset.test_dataset_name = main_args.dataset_name
    ossid_args.dataset.train_dataset_name = main_args.dataset_name

    # Keep all the zephyr results for the training set 
    ossid_args.dataset.zephyr_filter_key = None
    ossid_args.dataset.zephyr_results_percent = 1
    # use more templates for training
    ossid_args.dataset.train_local_template_sample_from = 10

    if main_args.n_local_test is not None:
        ossid_args.dataset.n_local_test = main_args.n_local_test
    elif main_args.use_pretrained_dtoid: # If their weights are used
        ossid_args.dataset.n_local_test = 160 
    else: # If our weights are used
        ossid_args.dataset.n_local_test = 10
    
    print("Number of local templates =", ossid_args.dataset.n_local_test)

    train_loader, valid_loader, test_loader = getDataloaders(ossid_args)

    # Sort the test loader
    test_loader.dataset.sortTargets(reverse=main_args.backward)

    ModelClass = DtoidNet
    model = DtoidNet(ossid_args)
    
    if main_args.use_pretrained_dtoid:
        # DTOID weightes provided by the authors will be loaded
        print("Loading DTOID weights provided by the original authors")
        pass
    elif ckpt_path is not None:
        print("Loading DTOID Model weights from", ckpt_path)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'])

    initial_state_dict = model.state_dict()

    model = model.to(0)
    model = model.eval()

    '''Initialize the trained Zephyr model'''
    if DATASET_NAME == 'lmo':
        CKPT_PATH = os.path.join(OSSID_CKPT_ROOT, "final_lmo.ckpt") # The path to the checkpoint
        USE_ICP = False # Not using ICP for LMO dataset, as it only uses PPF hypotheses, which are already after ICP processing. 
        MODEL_DATA_TPATH = os.path.join(OSSID_DATA_ROOT, "zephyr_model_data", "lmo", "model_cloud_{:02d}.npz") # path template to the sampled point cloud
        INCONST_RATIO_TH = 100
    elif DATASET_NAME == 'ycbv':
        if main_args.test_seen:
            CKPT_PATH_FOR_ODD = os.path.join(OSSID_CKPT_ROOT, "final_ycbv.ckpt")
            CKPT_PATH_FOR_EVEN = os.path.join(OSSID_CKPT_ROOT, "final_ycbv_valodd.ckpt")
        else:
            CKPT_PATH_FOR_ODD = os.path.join(OSSID_CKPT_ROOT, "final_ycbv_valodd.ckpt")
            CKPT_PATH_FOR_EVEN = os.path.join(OSSID_CKPT_ROOT, "final_ycbv.ckpt")
        USE_ICP = True # using ICP for LMO dataset
        MODEL_DATA_TPATH = os.path.join(OSSID_DATA_ROOT, "zephyr_model_data", "ycbv", "model_cloud_{:02d}.npz") # path template to the sampled point cloud
        INCONST_RATIO_TH = 10

    '''Set up the arguments for the model'''
    parser = getOptions()
    zephyr_args = parser.parse_args([])

    # Model-related
    zephyr_args.model_name = "pn2"
    zephyr_args.dataset = "HSVD_diff_uv_norm"
    zephyr_args.no_valid_proj = True
    zephyr_args.no_valid_depth = True
    zephyr_args.inconst_ratio_th = INCONST_RATIO_TH

    # Dataset-related
    zephyr_args.dataset_root = [""]
    zephyr_args.dataset_name = [DATASET_NAME]
    # zephyr_args.resume_path = CKPT_PATH
    zephyr_args.test_dataset = True

    '''Initialize pytorch dataloader and model'''
    # dataloader is only needed for the getPointNetData() function
    # zephyr_loader = getDataloader(zephyr_args)[0]
    zephyr_dataset = ScoreDataset([], "", DATASET_NAME, zephyr_args, mode='test')
    zephyr_args.dim_point = zephyr_dataset.dim_point
    zephyr_args.unseen_oids = []
    zephyr_args.extra_bottleneck_dim = 0

    if main_args.dataset_name == "ycbv":
        zephyr_model = PointNet2SSG(zephyr_args.dim_point, zephyr_args, num_class=1)
        zephyr_ckpt = torch.load(CKPT_PATH_FOR_ODD)
        zephyr_model.load_state_dict(zephyr_ckpt['state_dict'])
        zephyr_model = zephyr_model.to(0).eval()
        zephyr_model_for_odd = zephyr_model

        zephyr_model = PointNet2SSG(zephyr_args.dim_point, zephyr_args, num_class=1)
        zephyr_ckpt = torch.load(CKPT_PATH_FOR_EVEN)
        zephyr_model.load_state_dict(zephyr_ckpt['state_dict'])
        zephyr_model = zephyr_model.to(0).eval()
        zephyr_model_for_even = zephyr_model
    else:
        zephyr_model = PointNet2SSG(zephyr_args.dim_point, zephyr_args, num_class=1)
        zephyr_ckpt = torch.load(CKPT_PATH)
        zephyr_model.load_state_dict(zephyr_ckpt['state_dict'])
        zephyr_model = zephyr_model.to(0).eval()

    '''Initialize the BOP dataset'''
    # Set up the options
    bop_args = BopDatasetArgs(
        bop_root=BOP_DATASETS_ROOT, 
        dataset_name=DATASET_NAME, 
        model_type=None,
        split_name="bop_test", # This indicates we want to use the testing set defined in BOP challenge (different than original test set)
        split="test", 
        split_type=None, 
        ppf_results_file=None, 
        skip=1, # Iterate over all test samples, with no skipping
    )

    bop_dataset = BopDataset(bop_args)

    print("Length of the test dataset:", len(bop_dataset))

    '''Load the zephyr results'''
    zephyr_results = pickle.load(open(ossid_args.dataset.zephyr_result_path, 'rb'))
    zephyr_results = {(r['obj_id'], r['scene_id'], r['im_id']):r for r in zephyr_results}

    # Extract the training dataset from the training loader
    train_dtoid_bop_dataset = train_loader.dataset

    train_dtoid_bop_dataset.clearTargets()
    # Recover from the training/validation split on zephyr results
    train_dtoid_bop_dataset.zephyr_results = zephyr_results

    '''optimizer for dtoid model'''
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr = 1e-4, 
        weight_decay = 1e-6,
        amsgrad = True
    )

    '''Test the DTOID model before finetuning'''
    if main_args.raw_dtoid:
        print("Testing the DTOID model before finetuning")
        test_results = testDtoidModel(model, test_loader)

        save_path = os.path.join(SAVE_ROOT, "before_finetune_dtoid_results_%s.pkl" % main_args.exp_name)
        print("Saving results to", save_path)
        pickle.dump({
            "test_results": test_results,
            "main_args": main_args,
        }, open(save_path, 'wb'))

        df = pd.DataFrame.from_dict(test_results)
        print("DTOID mean IoU:", df['dtoid_iou'].mean())
        print("DTOID Valid IoU recall", (df['dtoid_iou'] > 0.5).astype(float).mean())

        return 0

    if main_args.use_sift_hypos:
        # Initialize the featured model for YCB-V dataset
        featured_objects = getFeaturizedModels(bop_dataset)

    '''main loop'''
    test_results = []
    finetune_logs = []

    renderers = {}

    # Create the surface model (PPF training stage)
    print("Creating PPF models using Halcon")
    ppf_models = {}
    for obj_id in bop_dataset.obj_ids:
        full_model_path = bop_dataset.model_tpath.format(obj_id=obj_id)
        if DATASET_NAME == 'ycbv':
            ppf_models[obj_id] = PPFModel(full_model_path, ModelSamplingDist = 0.03)
        else:
            ppf_models[obj_id] = PPFModel(full_model_path)

    # Preloading all model data
    print("Preloading all model data")
    model_data_all = {}
    for obj_id in bop_dataset.obj_ids:
        # Load the information of the model point cloud from the pre-processed dataset
        model_data_path = MODEL_DATA_TPATH.format(obj_id)
        model_data = np.load(model_data_path)
        model_points, model_colors, model_normals = model_data['model_points'], model_data['model_colors'], model_data['model_normals']
        model_data_all[obj_id] = (model_points, model_colors, model_normals)

    # The batch is the data for dtoid dataset
    for iteration, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        obj_id, scene_id, im_id = batch['obj_id'].item(), batch['scene_id'].item(), batch['im_id'].item()
        zr = zephyr_results[(obj_id, scene_id, im_id)]

        # Get the full mesh model provided by LineMOD dataset
        full_model_path = bop_dataset.model_tpath.format(obj_id=obj_id)

        # Get the raw data from the bop dataset, preparing for zephyr inference
        bop_data = bop_dataset.getDataByIds(obj_id, scene_id, im_id)
        
        # Extract the data from the bop datapoint
        img, depth, scene_camera = bop_data['img'], bop_data['depth'], bop_data['scene_camera']
        scene_meta = bop_data['scene_meta']
        mat_gt = bop_data['mat_gt']
        cam_K = np.asarray(scene_camera['cam_K']).reshape((3, 3))

        # Load the information of the model point cloud from the pre-processed dataset
        model_points, model_colors, model_normals = model_data_all[obj_id]

        # Get the proper error function according to whether the object is symmetric or not
        is_sym = obj_id in bop_dataset.sym_obj_ids
        
        if main_args.fast:
            err_func = add
        else:
            err_func = adi if is_sym else add

        # DTOID inference first
        dict_to(batch, 0)
        with torch.no_grad():
            model = model.eval()
            t1 = time.time()
            out = model.forwardTestTime(batch)
            time_dtoid = time.time() - t1

        final_bbox = to_np(out['final_bbox'][0])
        final_score = to_np(out['final_score'][0])
        dtoid_iou = to_np(out['seg_IoU'])
        dtoid_pred_mask = to_np(out['segmentation'][0,0])

        dtoid_confident = final_score[0] > DTOID_CONFIDENT_THRESHOLD

        use_dtoid_mask = False
        if main_args.ignore_dtoid_mask:
            use_dtoid_mask = False
        elif main_args.always_dtoid_mask:
            use_dtoid_mask = True
        else:
            use_dtoid_mask = dtoid_confident
        
        if iteration < main_args.finetune_warmup:
            use_dtoid_mask = False

        if not use_dtoid_mask:
            # Run zephyr on the whole image
            # Here we just get the stored zephyr results
            zephyr_score = zr['score']
            zephyr_mask = zr['pred_mask_visib']
            zephyr_pose = zr['pred_pose']
            
            pred_pose = to_np(zephyr_pose)
            pred_score = zephyr_score
            time_ppf = None
            time_sift = None
            time_zephyr = None
            time_icp = None
        else:
            # Take the prediction and run zephyr on the predicted mask

            # Get the mask according to dtoid detection results
            if main_args.use_dtoid_segmask:
                dtoid_mask = dtoid_pred_mask > 0.5 
                if dtoid_mask.sum() <= 25: # too few points
                    dtoid_mask = np.ones_like(dtoid_mask)
            else:
                dtoid_mask = np.zeros_like(depth)
                expand_ratio = 1.2
                img_h, img_w = depth.shape

                for i, (bbox, score) in enumerate(zip(final_bbox, final_score)):
                    # if main_args.always_dtoid_mask:
                    #     if i >= 1 and (dtoid_mask * (depth > 0).astype(int)).sum() > 0:
                    #         continue

                    # If the good boxes are already used and the mask is not empty
                    if score < 0.5 and (dtoid_mask * (depth > 0).astype(int)).sum() > 0:
                        continue

                    # Expand the detection bbox a bit
                    x1, y1, x2, y2 = bbox
                    x1, y1, x2, y2 = expandBox(x1, y1, x2, y2, img_h, img_w, expand_ratio)
                    dtoid_mask[int(y1):int(y2), int(x1):int(x2)] = 1

                # if dtoid_mask.sum() <= 50: # too few points
                #     dtoid_mask = np.ones_like(dtoid_mask)

            time_sift = 0
            # Get pose hypotheses
            if DATASET_NAME == 'ycbv':
                ppf_model = ppf_models[obj_id]

                # Run the PPF algorithm on the detected region
                scene_pc = depth2cloud(depth, np.logical_and(dtoid_mask, depth > 0), cam_K)
                # poses_ppf, scores_ppf, time_ppf = ppf_model.find_surface_model(scene_pc * 1000.0) # The wrapper requires the input to be in milimeters
                poses_ppf, scores_ppf, time_ppf = ppf_model.find_surface_model(scene_pc * 1000.0, DensePoseRefinement='false', SceneSamplingDist=0.03, RefPtRate=0.2) # The wrapper requires the input to be in milimeters
                poses_ppf[:, :3, 3] = poses_ppf[:, :3, 3] / 1000.0 # Convert from milimeter to meter
                
                poses_all = poses_ppf

                if main_args.use_sift_hypos:
                    t1 = time.time()
                    # Compute pose hypotheses from SIFT feature matches
                    try:
                        keypoints, features, cloud, frames = featurizeScene(img, depth_im_to_dist_im_fast(depth, cam_K), dtoid_mask, scene_meta, [11], [11])
                    except ValueError:
                        # The mask is too small to get any SIFT features
                        time_sift = None
                        poses_sift = np.stack([np.eye(4) for _ in range(20)], axis=0)
                        poses_all = np.concatenate([poses_sift, poses_all], axis=0)
                    else:
                        '''Match to corresponding object'''
                        poses_sift, match_aux = featured_objects[obj_id].match(features, frames, mat_gt)
                        time_sift = time.time() - t1
                        poses_all = np.concatenate([poses_sift, poses_all], axis=0)
                
                # Shift the model points from YCB-V dataset to BOP
                model_points = modelPointsShiftYcbv2Bop(model_points, obj_id)
            else:
                ppf_model = ppf_models[obj_id]

                # Run the PPF algorithm on the detected region
                scene_pc = depth2cloud(depth, np.logical_and(dtoid_mask, depth > 0), cam_K)
                poses_ppf, scores_ppf, time_ppf = ppf_model.find_surface_model(scene_pc * 1000.0) # The wrapper requires the input to be in milimeters
                poses_ppf[:, :3, 3] = poses_ppf[:, :3, 3] / 1000.0 # Convert from milimeter to meter
                
                poses_all = poses_ppf

            # Recompute the per-point error for newly-estimated poses
            pp_err = np.asarray([err_func(mat[:3,:3], mat[:3, 3], mat_gt[:3, :3], mat_gt[:3, 3], model_points) for mat in poses_all])

            # Run zephyr
            data_for_zephyr = {
                "img": img, "depth": depth, "cam_K": cam_K, 
                "model_colors": model_colors, "model_points": model_points, "model_normals":model_normals, 
                "pose_hypos": poses_all, 'pp_err': pp_err
            }

            if main_args.dataset_name == 'ycbv':
                # Handle two models for YCB-V
                zephyr_model = zephyr_model_for_even if obj_id % 2 == 0 else zephyr_model_for_odd
            poses_zephyr, scores_zephyr, pp_err, uv_original, time_zephyr = networkInference(zephyr_model, zephyr_dataset, data_for_zephyr, return_time=True)
            
            pred_score = scores_zephyr.max().item()
            pred_idx = scores_zephyr.argmax()
            pred_pose = poses_zephyr[pred_idx]
            pred_err = pp_err[pred_idx].item()

            # Run ICP as a post-processing step
            time_icp = 0
            if USE_ICP:
                uv_original = to_np(uv_original)
                t1 = time.time()
                pred_pose, _  = icpRefinement(
                    depth, uv_original[pred_idx],
                    pred_pose, cam_K, model_points, inpaint_depth=False, icp_max_dist=0.01
                )
                time_icp = time.time() - t1

        pred_err = err_func(pred_pose[:3,:3], pred_pose[:3, 3], mat_gt[:3, :3], mat_gt[:3, 3], model_points)

        # Render the object to get predicted color and depth
        if obj_id not in renderers:
            renderer = Renderer(K2meta(cam_K))
            renderer.addObject(obj_id, full_model_path, pose=pred_pose, mm2m=True, simplify=main_args.fast)
            renderers[obj_id] = renderer
        else:
            renderer = renderers[obj_id]
            renderer.obj_nodes[obj_id].matrix = pred_pose

        pred_color, pred_depth = renderer.render(depth_only=True)

        # Compute the IoU metrics
        pred_mask = pred_depth > 0
        gt_mask = bop_data['mask_gt'] > 0
        gt_mask_visib = bop_data['mask_gt_visib'] > 0

        pred_mask_visib = estimate_visib_mask_gt(depth, pred_depth, 15/1000.)

        # finetune DTOID after every a certain number of datapoints are added
        if main_args.use_oracle_gt:
            zephyr_confident = True
        else:
            zephyr_confident = pred_score > ZEPHYR_CONFIDENT_THRESHOLD
            
        finetune = False
        time_finetune = 0
        if not main_args.no_finetune and zephyr_confident:
            # Add the datapoint into the finetuning dataset
            train_dtoid_bop_dataset.addTarget(obj_id, scene_id, im_id)
            if main_args.use_oracle_gt:
                train_dtoid_bop_dataset.updateZephyrMask(obj_id, scene_id, im_id, gt_mask_visib, pred_score)
            else:
                train_dtoid_bop_dataset.updateZephyrMask(obj_id, scene_id, im_id, pred_mask_visib, pred_score)
            if len(train_dtoid_bop_dataset) == next_finetune_number:
                finetune = True

                if main_args.finetune_reset:
                    print("Resetting the DTOID weights, and the optimizer")
                    model.load_state_dict(initial_state_dict)
                    optimizer = torch.optim.Adam(
                        model.parameters(), 
                        lr = 1e-4, 
                        weight_decay = 1e-6,
                        amsgrad = True
                    )

                print("Starting finetuning DTOID at iteration %d" % iteration)
                t1 = time.time()
                model, train_logs = finetuneDtoid(model, train_dtoid_bop_dataset, optimizer, epochs=main_args.finetune_epochs, batch_size=args.finetune_batch_size)
                time_finetune = time.time() - t1

                if main_args.save_each:
                    # save the model weights immediately after finetuning
                    model_save_folder = os.path.join(SAVE_ROOT, main_args.exp_name)
                    makeFolder(model_save_folder)

                    model_save_path = os.path.join(model_save_folder, "epoch_%d.ckpt" % iteration)
                    print("Saving the current model at", model_save_path)
                    torch.save({
                        "iteration": iteration,
                        "model_state_dict": model.state_dict(),
                        "conf": ossid_args,
                    }, model_save_path)

                finetune_logs.append(train_logs)

                if main_args.non_cum:
                    print("Clearing finetuning targets")
                    train_dtoid_bop_dataset.clearTargets()
                    next_finetune_number = main_args.finetune_interval
                else:
                    next_finetune_number = next_finetune_number + main_args.finetune_interval

        iou = np.logical_and(pred_mask, gt_mask).sum().astype(float) / np.logical_or(pred_mask, gt_mask).sum().astype(float)
        iou_visib = np.logical_and(pred_mask_visib, gt_mask_visib).sum().astype(float) / np.logical_or(pred_mask_visib, gt_mask_visib).sum().astype(float)

        result = {}
        result['obj_id'] = obj_id
        result['scene_id'] = scene_id
        result['im_id'] = im_id

        result['dtoid_confident'] = dtoid_confident
        result['zephyr_confident'] = zephyr_confident
        result['use_dtoid_mask'] = use_dtoid_mask
        result['finetune'] = finetune

        result['dtoid_iou'] = dtoid_iou
        result['dtoid_pred_mask'] = dtoid_pred_mask
        result['dtoid_bbox'] = final_bbox
        result['dtoid_score'] = final_score

        result['pred_pose'] = to_np(pred_pose)
        result['pred_score'] = pred_score
        result['pred_err'] = pred_err
        result['pred_add01d'] = float(pred_err < 0.1 * OBJECT_DIAMETERES[DATASET_NAME][obj_id])
        result['pred_mask'] = pred_mask
        result['pred_mask_visib'] = pred_mask_visib
        result['pred_iou'] = iou
        result['pred_iou_visib'] = iou_visib

        result['time_dtoid'] = time_dtoid
        result['time_ppf'] = time_ppf
        result['time_sift'] = time_sift
        result['time_zephyr'] = time_zephyr
        result['time_icp'] = time_icp
        result['time_finetune'] = time_finetune

        test_results.append(result)

    save_path = os.path.join(SAVE_ROOT, "results_%s.pkl" % main_args.exp_name)
    print("Saving results to", save_path)
    pickle.dump({
        "test_results": test_results,
        "main_args": main_args,
        "finetune_logs": finetune_logs,
        "final_state_dict": model.state_dict(),
    }, open(save_path, 'wb'))

    print("Saving results in BOP format")
    saveResultsBop(
        test_results, 
        BOP_RESULTS_FOLDER, "online-%s" % main_args.exp_name, 
        main_args.dataset_name, pose_key='pred_pose', score_key='pred_score',
        run_eval_script=True,
    )

    df = pd.DataFrame.from_dict(test_results)
    print("DTOID mean IoU:", df['dtoid_iou'].mean())
    print("DTOID Valid IoU recall", (df['dtoid_iou'] > 0.5).astype(float).mean())
    print("Zephyr Valid IoU recall", (df['pred_iou_visib'] > 0.5).astype(float).mean())

    '''Evaluate the results in terms of Detection mAP'''
    tmp_root = Path(OSSID_DET_ROOT)
    saveLmoYcbvGT(tmp_root, bop_root=BOP_DATASETS_ROOT)
    evalFinetuneResults(save_path, DATASET_NAME, tmp_root)

def testDtoidModel(model, test_loader):
    '''
    A function performing test epoch on the test_loader
    '''
    test_results = []
    for batch in tqdm(test_loader):
        obj_id, scene_id, im_id = batch['obj_id'].item(), batch['scene_id'].item(), batch['im_id'].item()
        dict_to(batch, 0)
        with torch.no_grad():
            model = model.eval()
            out = model.forwardTestTime(batch)

        final_bbox = to_np(out['final_bbox'][0])
        final_score = to_np(out['final_score'][0])
        dtoid_iou = to_np(out['seg_IoU'])
        dtoid_pred_mask = to_np(out['segmentation'][0,0])
        
        result = {}
        result['obj_id'] = obj_id
        result['scene_id'] = scene_id
        result['im_id'] = im_id
        result['dtoid_bbox'] = final_bbox
        result['dtoid_score'] = final_score
        result['dtoid_iou'] = dtoid_iou
        result['dtoid_pred_mask'] = dtoid_pred_mask
        result['gt_bbox'] = to_np(batch['bbox_gt'][0,0,:4])
        test_results.append(result)

    return test_results

def finetuneDtoid(model, train_dataset, optimizer, epochs=1, batch_size=8):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=8, collate_fn = collate_fn, 
        shuffle=True, pin_memory=True
    )

    model = model.train()

    train_logs = []
    for epoch in range(epochs):
        epoch_logs = []
        for batch in tqdm(train_loader):
            batch = move_to(batch, 0)
            if type(model) is MaskRCNN:
                out = model(*batch)
            else:
                out = model(batch)

            loss = out['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_logs.append({
                "train_loss": loss.item()
            })
        train_logs.append(epoch_logs)

    return model, train_logs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for test-time training')
    parser.add_argument("--dataset_name", type=str, default='lmo', choices=['lmo', 'ycbv'], help="The name of the dataset to be used")
    parser.add_argument("--exp_name", type=str, default="exp", help="The name of the experiement to be appended to the saved result file")

    # About which initial weight to use
    # By default, the DTOID pretrained on the render dataset will be loaded as the initial weight
    parser.add_argument("--use_offline_model", action="store_true", help="If set, the DTOID model already finetuned offline will be loaded. ")
    parser.add_argument("--use_pretrained_dtoid", action="store_true", help="If True, the DTOID model provided by the authors will be used. ")
    parser.add_argument("--dtoid_weights_path", type=str, default=None, help="If not None, DTOID weights will be loaded from this path and override all other arguments. ")
    parser.add_argument("--n_local_test", type=int, default=None, help="If not None, this value will be used as the number of local templates used for inference. ")
    parser.add_argument("--use_dtoid_segmask", action="store_true", help="If set, the segmentation mask by DTOID will be used. Otherwise the bbox will be used. ")
    parser.add_argument("--ignore_dtoid_mask", action="store_true", help="If set, the zephyr will not be run on the DTOID mask, but on the entire image. ")
    parser.add_argument("--always_dtoid_mask", action="store_true", help="If set, the confidence filtering by DTOID scores will be turned off. ")
    parser.add_argument("--use_oracle_gt", action="store_true", help="If set, the ground truth mask and box instead of zephyr's will be used to finetune the DTOID. ")

    parser.add_argument("--use_sift_hypos", action="store_true", help="If set, the pose hypotheses will also be estimated from SIFT feature matches. ")
    parser.add_argument("--test_seen", action="store_true", help="If set, the models trained on the same object set will be use for testing. ")
    parser.add_argument("--backward", action="store_true", help="If set, the images will be sorted in the backward image ID order. ")

    parser.add_argument("--use_maskrcnn", action="store_true", help='If set, the mask rcnn model will be used')

    # Detailed parameters for finetuning
    parser.add_argument("--finetune_interval", type=int, default=8, help="Finetuning will happen after every this number of finetuning datapoints are added")
    parser.add_argument("--finetune_warmup", type=int, default=0, help="Finetuning will happen only after this number of datapoints are added")
    parser.add_argument("--finetune_epochs", type=int, default=1, help="The epochs of training at each time DTOID is finetuned")
    parser.add_argument("--finetune_reset", action="store_true", help="If set, before each finetuning, the network will be reset to the initial weights")
    parser.add_argument("--finetune_batch_size", type=int, default=8, help="The batch size for finetuning. ")
    parser.add_argument("--non_cum", action="store_true", help="If set, the finetuning example will be cleared after finetuning. ")
    parser.add_argument("--save_each", action="store_true", help="If set, the weights of the model will be saved after each finetuning function")
    
    # About how to test the DTOID
    # By default, the DTOID will be finetuned and tested gradually
    parser.add_argument("--raw_dtoid", action="store_true", help="If True, the DTOID model before finetuning will be tested")
    parser.add_argument("--no_finetune", action="store_true", help="If set, the DTOID will not be finetuned. This will be a test script for DTOID+Zephyr")

    parser.add_argument("--fast", action="store_true", help="If set, the script will be run at a fast mode. (only add will be used)")
     
    args = parser.parse_args()
    main(args)