#!/usr/bin/env python3
"""Baseline methods

Description:
    Optical flow based superpixel propagation

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
from pathlib import Path

# 3rd party libs
import numpy as np
from skimage.measure import regionprops

# local files
ROOT_DIR = Path(__file__).resolve().parents[1]
BORUVKA_DIR = ROOT_DIR / 'external' / 'boruvka-superpixel' / 'pybuild'
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BORUVKA_DIR))

import boruvka_superpixel
from tools.utils import average


def baseline(img_ins: np.ndarray,
             gt_ins: np.ndarray,
             of_ins: np.ndarray,
             edge_ins: np.ndarray,
             n_supix: int = 512) -> np.ndarray:
    """Optical flow based pixel (centroid) propagation

    This method...
        calculate superpixel segmentations
        translate superpixel centroids
        calculate output masks

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
        edge_ins (np.ndarray): edge images of a single video 
                             as a 3D numpy array:
                             shape=(height, width, num of frames)
                             calculated with HED using the input images
        n_supix (int): number of superpixels

    Returns:
        (np.ndarray): output masks as a 3D numpy array

    """
    h, w, l = img_ins.shape[:3]
    zero_edge = np.zeros(shape=(h, w), dtype=np.uint8)
    label_outs = np.zeros(shape=(h, w, l), dtype=np.int16)
    masks = np.zeros(shape=(h, w, l), dtype=np.uint8)
    masks[:,:,0] = gt_ins[:,:,0]

    # boruvka's algorithm
    for i in range(l):
        bosupix = boruvka_superpixel.BoruvkaSuperpixel()
        label_out = np.zeros(shape=(h, w, 1), dtype=np.int16)
        features = img_ins[:,:,i,:]
        bosupix.build_2d(features, edge_ins[:,:,i].astype(features.dtype), edgeless=zero_edge)
        bosupix.label_o(n_supix, label_out)
        label_out = np.squeeze(label_out, axis=2)
        label_outs[:,:,i] = label_out

    for i in range(l-1):
        mask_sp_ids = np.unique(label_outs[masks[:,:,i]>0,i])
        of_avg = average(of_ins[:,:,i], label_outs[:,:,i])

        for mask_sp_id in mask_sp_ids:
            cx, cy = np.where(label_outs[:,:,i] == mask_sp_id)
            cx = np.rint(np.mean(cx)).astype(np.int16)
            cy = np.rint(np.mean(cy)).astype(np.int16)

            of = of_avg[cx,cy]
            x = cx + np.rint(of[0]).astype(np.int16)
            y = cy + np.rint(of[1]).astype(np.int16)

            if x < 0: x = 0
            if x > h - 1: x = h - 1
            if y < 0: y = 0
            if y > w - 1: y = w - 1

            masks[label_outs[:,:,i+1] == label_outs[x,y,i+1],i+1] = 255

    return masks


def baseline2(img_ins: np.ndarray,
              gt_ins: np.ndarray,
              of_ins: np.ndarray,
              edge_ins: np.ndarray,
              n_supix: int = 512,
              overlap: float = 0.5) -> np.ndarray:
    """Optical flow based superpixel propagation

    This method...
        calculate superpixel segmentations
        translate superpixels
        calculate output masks

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
        edge_ins (np.ndarray): edge images of a single video 
                             as a 3D numpy array:
                             shape=(height, width, num of frames)
                             calculated with HED using the input images
        n_supix (int): number of superpixels
        overlap (float): accepted overlap of superpixels

    Returns:
        (np.ndarray): output masks as a 3D numpy array

    """
    h, w, l = img_ins.shape[:3]
    zero_edge = np.zeros(shape=(h, w), dtype=np.uint8)
    label_outs = np.zeros(shape=(h, w, l), dtype=np.int16)
    masks = np.zeros(shape=(h, w, l), dtype=np.uint8)
    masks[:,:,0] = gt_ins[:,:,0]

    for i in range(l-1):
        # boruvka's algorithm
        bosupix = boruvka_superpixel.BoruvkaSuperpixel()
        label_out = np.zeros(shape=(h, w, 1), dtype=np.int16)
        features = img_ins[:,:,i,:]
        bosupix.build_2d(features, edge_ins[:,:,i].astype(features.dtype), edgeless=zero_edge)
        bosupix.label_o(n_supix, label_out)
        label_out = np.squeeze(label_out, axis=2)
        label_outs[:,:,i] = label_out

        # select superpixels based on a predicted or gt mask
        mask_sp_ids = []
        sp_ids = np.unique(label_out[masks[:,:,i] > 0])

        for sp_id in sp_ids:
            img_area = np.sum(label_out == sp_id)
            mask_area = np.sum(np.logical_and(label_out == sp_id, masks[:,:,i] > 0))

            if float(mask_area) / float(img_area) > overlap:
                mask_sp_ids.append(sp_id)

        # translate superpixels with mean OF
        of_avg = average(of_ins[:,:,i,:], label_out)
        pred = np.zeros(shape=(h, w), dtype=np.uint8)

        for label in np.unique(mask_sp_ids):
            l_x, l_y = np.where(label_out == label) # mean OF of the selected superpixel
            
            of = of_avg[l_x[0], l_y[0]]
            sp_x = l_x + np.rint(of[0]).astype(np.int16)
            sp_y = l_y + np.rint(of[1]).astype(np.int16)

            sp_x[sp_x < 0] = 0
            sp_x[sp_x > h-1] = h-1
            sp_y[sp_y < 0] = 0
            sp_y[sp_y > w-1] = w-1

            masks[l_x, l_y, i] = 255            # IoU selected sp mask for t
            pred[sp_x, sp_y] = 255              # OF translated sp mask for t+1

        masks[:,:,i+1] = pred                   # mask for t+1

    return masks


