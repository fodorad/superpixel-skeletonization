#!/usr/bin/env python3
"""Single-skeleton method

Description:
    Optical flow based mask skeleton propagation

Paper:
    Adam Fodor, Aron Fothi, Laszlo Kopacsi, Ellak Somfai, and Andras Lorincz
    Skeletonization Combined with Deep Neural Networks for Superpixel Temporal Propagation
    International Joint Conference on Neural Networks (IJCNN), 2019

Contact:
    Ádám Fodor -- foauaai@inf.elte.hu
"""
# pylint: disable=E1101
# pylint: disable=E0401
# pylint: disable=E0611

# standard libs
import sys
from typing import Tuple
from pathlib import Path

# 3rd party libs
import cv2
import numpy as np
from skimage.morphology import medial_axis
from skimage import img_as_bool, img_as_ubyte
from skimage._shared._warnings import expected_warnings

# local files
ROOT_DIR = Path(__file__).resolve().parents[1]
BORUVKA_DIR = ROOT_DIR / 'external' / 'boruvka-superpixel' / 'pybuild'
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BORUVKA_DIR))

import boruvka_superpixel
from tools.utils import average
from tools.utils import computeImg as computeFlowImg


def single_skeleton(img_ins: np.ndarray,
                    gt_ins: np.ndarray,
                    of_ins: np.ndarray,
                    depth_ins: np.ndarray,
                    edge_ins: np.ndarray,
                    input_id: int = 0,
                    n_supix: int = 512,
                    overlap: float = 0.5,
                    disk_thr: float = 0,
                    skeleton_thr: float = 0.2) -> np.ndarray:
    """Optical flow based mask skeleton propagation

    This method...
        calculate superpixel segmentations
        calculate and translate medial axis transform

    Args:
        img_ins (np.ndarray): input images of a single video 
                              as a 4D numpy array:
                              shape=(height, width, num of frames, num of channels)
        gt_ins (np.ndarray): annotation images of a single video 
                             as a 3D numpy array:
                             shape=(height, width, num of frames)
        of_ins (np.ndarray): optical flow images of a single video 
                             as a 4D numpy array:
                             shape=(height, width, num of frames, len of vectors)
                             calculated with PWC-Net using the input images
        depth_ins (np.ndarray): depth images of a single video 
                             as a 3D numpy array:
                             shape=(height, width, num of frames)
                             calculated with MegaDepth using the input images
        edge_ins (np.ndarray): edge images of a single video 
                             as a 3D numpy array:
                             shape=(height, width, num of frames)
                             calculated with HED using the input images
        input_id (int): input feature configuration id
        n_supix (int): number of superpixels
        overlap (float): accepted overlap of superpixels
        disk_thr (float): filter

    Returns:
        (np.ndarray): output masks as a 3D numpy array

    """
    h, w, l = img_ins.shape[:3]
    zero_edge = np.zeros(shape=(h, w), dtype=np.uint8)
    label_outs = np.zeros(shape=(h, w, l), dtype=np.int16)
    shifted_skeleton = np.zeros(shape=(h, w), dtype=np.int16)
    masks = np.zeros(shape=(h, w, l), dtype=np.uint8)
    masks[:,:,0] = gt_ins[:,:,0]

    for i in range(l-1):

        # boruvka's algorithm
        bosupix = boruvka_superpixel.BoruvkaSuperpixel()
        label_out = np.zeros(shape=(h, w, 1), dtype=np.int16)

        # prepare inputs considering different configurations
        if input_id == 0: # color + depth + hed
            with expected_warnings(['precision loss']):
                depth_img = img_as_ubyte(depth_ins[:,:,i,None])
            features = np.concatenate([img_ins[:,:,i,:], depth_img], axis=-1)
            edges = edge_ins[:,:,i]
        elif input_id == 1: # hed
            features = edge_ins[:,:,i]
            edges = zero_edge
        elif input_id == 2: # color
            features = img_ins[:,:,i,:]
            edges = zero_edge
        elif input_id == 3: # color + hed
            features = img_ins[:,:,i,:]
            edges = edge_ins[:,:,i]
        elif input_id == 4: # depth
            with expected_warnings(['precision loss']):
                depth_img = img_as_ubyte(depth_ins[:,:,i,None])
            features = depth_img
            edges = zero_edge
        elif input_id == 5: # depth + hed
            with expected_warnings(['precision loss']):
                depth_img = img_as_ubyte(depth_ins[:,:,i,None])
            features = depth_img
            edges = edge_ins[:,:,i]
        elif input_id == 6: # of (as img)
            features = computeFlowImg(of_ins[:,:,i,:])
            edges = zero_edge
        elif input_id == 7: # of (as img) + hed
            features = computeFlowImg(of_ins[:,:,i,:])
            edges = edge_ins[:,:,i]
        else:
            raise NotImplementedError("Invalid input_id option.")

        bosupix.build_2d(features, edges.astype(features.dtype))
        bosupix.label_o(n_supix, label_out)
        label_out = np.squeeze(label_out, axis=2)
        label_outs[:, :, i] = label_out

        # select superpixels based on a predicted or gt mask
        mask_sp_ids = []
        sp_ids = np.unique(label_out[masks[:,:,i] > 0])

        for sp_id in sp_ids:
            sp_area = np.sum(label_out == sp_id)
            mask_area = np.sum(np.logical_and(label_out == sp_id, masks[:,:,i] > 0))

            if float(mask_area) / float(sp_area) > overlap:
                mask_sp_ids.append(sp_id)

        if i > 0:
            masks[:,:,i] = 0
            for mask_sp_id in mask_sp_ids:
                masks[label_out == mask_sp_id,i] = 255

        # calculate medial axis transform for the mask
        skel, dist = calculate_skeleton(masks[:,:,i], is_binary=False)

        # weigth skeleton with the distance, then scale (normalize) between [0, 255]
        dskel = dist * skel

        # apply skeleton threshold
        dskel_ip = np.interp(dskel, (dskel.min(), dskel.max()), (0, 255)).astype(np.int16)
        dskel[dskel_ip < 255 * skeleton_thr] = 0

        # shift mask skeleton with their own pixel-wise optical flow
        shifted_skeleton[:, :] = 0
        w_sp_x, w_sp_y = np.where(dskel > 0)
        border = w
        reconst_mask = np.zeros(shape=(h+2*border, w+2*border), dtype=np.int16)

        of = of_ins[w_sp_x,w_sp_y,i,:]
        sp_x = w_sp_x + np.rint(of[:,0]).astype(np.int16)
        sp_y = w_sp_y + np.rint(of[:,1]).astype(np.int16)
        sp_x[sp_x < 0] = 0
        sp_x[sp_x > h-1] = h-1
        sp_y[sp_y < 0] = 0
        sp_y[sp_y > w-1] = w-1

        shifted_skeleton[sp_x, sp_y] = dskel[w_sp_x, w_sp_y]

        for ind in range(len(sp_x)):
            c_x = sp_x[ind]
            c_y = sp_y[ind]
            reconst_part = np.zeros(shape=(h + 2 * border, w + 2 * border), dtype=np.uint8)

            try:
                cv2.circle(reconst_part, (c_y + border, c_x + border), shifted_skeleton[c_x, c_y], 1, -1)
            except Exception:
                # ignore exceptions like drawing circles that outflow 
                # the (h + 2 * border) x (w + 2 * border) canvas
                pass
            
            reconst_mask += reconst_part

        # draw reconstruated mask
        reconst_mask = reconst_mask[border:(h+border), border:(w+border)]

        # disk part that lower than threshold is background
        if disk_thr != 0:
            reconst_mask[reconst_mask <= disk_thr] = 0

        masks[reconst_mask > 0,i+1] = 255

    return masks


def calculate_skeleton(mask: np.ndarray,
                       is_binary: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    '''Calculate Medial Axis Transform

    Args:
        mask (np.ndarray): binary image (or values with 0 and 255)
        is_binary_mask (bool) [True]: the mask is binary or not

    Returns:
        (np.ndarray, np.ndarray): 
            skeleton as a bool numpy array
            distances as a numpy array

    '''

    if ~is_binary:
        with expected_warnings(['precision loss']):
            mask = img_as_bool(mask)

    skel, distance = medial_axis(mask, return_distance=True)
    
    return skel, distance


