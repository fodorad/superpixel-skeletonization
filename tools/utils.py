#!/usr/bin/env python3
"""Utility functions

Description:
    Collection of useful functions

"""
# pylint: disable=E1101
# pylint: disable=E0401

# standard libs
import sys
from typing import List

# 3rd party libs
import numpy as np


def average(data: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Calculate mean values using superpixel labels.

    Args:
        data: features (e.g. RGB)
        label: labels produced by Boruvka’s Algorithm
    
    Returns:
        Mean feature values within superpixels

    """
    sp_ids = np.unique(label)
    avg = np.zeros_like(data)

    for sp_id in sp_ids:
        avg[label==sp_id] = np.mean(data[label==sp_id], axis=0)
    return avg


def get_random_color() -> List[int]:
    """Generates a random RGB triple (between 0-255)

    Returns:
        (List[int]): RGB triple

    """
    return list(np.random.choice(range(256), size=3))


def random_colors(N: int) -> List[List[int]]:
    """Generates N random RGB triples

    Returns:
        (List[List[int]]): List of RGB triples

    """
    colors = np.unique([get_random_color() for _ in range(N)], axis=0).tolist()
    while len(colors) < N:
        colors += [get_random_color() for _ in range(N - len(colors))]
        colors = np.unique(colors, axis=0).tolist()
    return colors


def computeImg(flow: np.ndarray) -> np.ndarray:
    """Compute colored image to visualize optical flow file .flo

    Source:
        According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein  
        Contact: dqsun@cs.brown.edu
        Contact: schar@middlebury.edu

        Author: Johannes Oswald, Technical University Munich
        Contact: johannes.oswald@tum.de
        Date: 26/04/2017

        For more information, check http://vision.middlebury.edu/flow/ 
    
    Args:
        flow (np.ndarray): 2D optical flow as numpy array
    
    Returns:
        (np.ndarray): optical flow as a 3-channel image

    """
    eps = sys.float_info.epsilon

    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10 

    u = flow[: , : , 0]
    v = flow[: , : , 1] 

    maxu = -999
    maxv = -999 
    minu = 999
    minv = 999  
    maxrad = -1

    #fix unknown flow
    greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
    greater_v = np.where(v > UNKNOWN_FLOW_THRESH)

    u[greater_u] = 0
    u[greater_v] = 0
    v[greater_u] = 0 
    v[greater_v] = 0    
    
    maxu = max([maxu, np.amax(u)])
    minu = min([minu, np.amin(u)])  
    maxv = max([maxv, np.amax(v)])
    minv = min([minv, np.amin(v)])
    
    rad = np.sqrt(np.multiply(u,u)+np.multiply(v,v)) 
    maxrad = max([maxrad, np.amax(rad)])
    print("""max flow: %.4f 
             flow range: 
             u = %.3f .. %.3f; 
             v = %.3f .. %.3f\n""" % (maxrad, minu, maxu, minv, maxv)) 
    
    u = u / (maxrad + eps)
    v = v / (maxrad + eps)
    img = computeColor(u, v)
    
    return img


def makeColorwheel() -> np.ndarray:
    """Color encoding scheme

    Source:
	    adapted from the color circle idea described at
	    http://members.shaw.ca/quadibloc/other/colint.htm

    Returns:
        (np.ndarray): color wheel

    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6  
    ncols = RY + YG + GC + CB + BM + MR 
    colorwheel = np.zeros([ncols, 3]) # r g b   
    col = 0

    #RY
    colorwheel[0:RY,0] = 255
    colorwheel[0:RY,1] = np.floor(255 * np.arange(0, RY, 1) / RY)
    col += RY   
    
    #YG
    colorwheel[col:YG+col,0] = 255 - np.floor(255 * np.arange(0, YG, 1) / YG)
    colorwheel[col:YG+col,1] = 255
    col += YG

    #GC
    colorwheel[col:GC+col,1]= 255 
    colorwheel[col:GC+col,2] = np.floor(255 * np.arange(0, GC, 1) / GC)
    col += GC

    #CB
    colorwheel[col:CB+col,1]= 255 - np.floor(255 * np.arange(0, CB, 1) / CB)
    colorwheel[col:CB+col,2] = 255
    col += CB
 
    #BM
    colorwheel[col:BM+col,2]= 255 
    colorwheel[col:BM+col,0] = np.floor(255 * np.arange(0, BM, 1) / BM)
    col += BM

    #MR
    colorwheel[col:MR+col,2]= 255 - np.floor(255 * np.arange(0, MR, 1) / MR)
    colorwheel[col:MR+col,0] = 255

    return 	colorwheel  


def computeColor(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute colored image

    Args:
        u (np.ndarray): optical flow component
        v (np.ndarray): optical flow component
    
    Returns:
        (np.ndarray): optical flow as a 3-channel image

    """
    colorwheel = makeColorwheel()

    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    nan_u = np.where(nan_u)
    nan_v = np.where(nan_v)   

    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0 
    v[nan_v] = 0    

    ncols = colorwheel.shape[0]
    radius = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) /2 * (ncols-1) # -1~1 maped to 1~ncols
    k0 = fk.astype(np.uint8)  # 1, 2, ..., ncols
    k1 = k0+1
    k1[k1 == ncols] = 0
    f = fk - k0 

    img = np.empty([k1.shape[0], k1.shape[1],3])
    ncolors = colorwheel.shape[1]

    for i in range(ncolors):
        tmp = colorwheel[:,i]
        col0 = tmp[k0]/255
        col1 = tmp[k1]/255
        col = (1-f)*col0 + f*col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx]*(1-col[idx]) # increase saturation with radius    
        col[~idx] *= 0.75 # out of range
        img[:,:,2-i] = np.floor(255*col).astype(np.uint8) 

    return img.astype(np.uint8) 


