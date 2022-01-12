#!/usr/bin/env python3
#
# Under refactoring.
# Please use the currently best published single_skeleton method instead
#

# standard lib
import sys, traceback
import timeit
import warnings
import argparse
from os import listdir, makedirs
from pathlib import Path
from os.path import isfile, isdir, join, exists, basename, dirname
from multiprocessing import Pool

# numpy family
import numpy as np

# 3rd party
import random
import cv2
import skimage
import skimage.feature
from skimage.measure import regionprops
from skimage import img_as_bool, io
from skimage.transform import resize
from skimage.morphology import medial_axis, thin, skeletonize, skeletonize_3d

# local
ROOT_DIR = Path(__file__).resolve().parents[1]
BORUVKA_DIR = ROOT_DIR / 'external' / 'boruvka-superpixel' / 'pybuild'
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BORUVKA_DIR))
import boruvka_superpixel
from tools.utils import computeImg

start = timeit.default_timer()
random.seed = 6666
debug_mode = True


def average(data, label):

    sp_ids = np.unique(label)
    avg = np.zeros_like(data)

    for sp_id in sp_ids:
        avg[label==sp_id] = np.mean(data[label==sp_id], axis=0)
    return avg


def get_colors(n):
  ret = []
  r = int(random.random() * 256)
  g = int(random.random() * 256)
  b = int(random.random() * 256)
  step = 256 / n
  for i in range(n):
    r += step
    g += step
    b += step
    r = int(r) % 256
    g = int(g) % 256
    b = int(b) % 256
    ret.append((r,g,b))
  return ret


def calculate_skeleton(mask: np.ndarray,
                       is_binary: bool = True) -> np.ndarray:
    '''Calculate Medial Axis Transform

    Args:
        mask (np.ndarray): binary image (or values with 0 and 255)
        is_binary_mask (bool) [True]: the mask is binary or not
    '''

    # ignore uint -> bool conversion warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if ~is_binary:
           mask = img_as_bool(mask)
        skel, distance = medial_axis(mask, return_distance=True)

    return skel, distance


def calculate_thinning(mask: np.ndarray,
                       is_binary: bool = True) -> np.ndarray:
    '''Calculate Thinning

    Args:
        mask (np.ndarray): binary image (or values with 0 and 255)
        is_binary_mask (bool) [True]: the mask is binary or not
    '''

    # ignore uint -> bool conversion warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if ~is_binary:
           mask = img_as_bool(mask)
        skel = thin(mask)

    return skel


def boruvkasupix2D(params):

    infolder, maskfolder, offolder, edgefolder, depthfolder, outfolder, technique, sp_mask_ratio, n_supix, mat_thr = params

    onlyfiles = [f for f in listdir(infolder) if isfile(join(infolder, f))]
    onlyfiles.sort()

    outfolder_skeleton = join(dirname(outfolder) + "_" + str(n_supix) + "_" + str(sp_mask_ratio) + "_skel", basename(outfolder))
    outfolder_segmentation = join(dirname(outfolder) + "_" + str(n_supix) + "_" + str(sp_mask_ratio) + "_" + str(mat_thr), basename(outfolder))
    outfolder_test = join(dirname(outfolder) + "_" + str(n_supix) + "_" + str(sp_mask_ratio) + "_" + str(mat_thr) + "_test", basename(outfolder))

    if not exists(outfolder_skeleton):
        makedirs(outfolder_skeleton)

    if not exists(outfolder_segmentation):
        makedirs(outfolder_segmentation)

    if not exists(outfolder_test):
        makedirs(outfolder_test)

    first_mask = io.imread(join(maskfolder, '00000.png'))

    mask = first_mask
    mask_skel_ids = []
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    max_height, max_width = mask.shape

    mask_skel_ids = []
    label_out = []

    colors = get_colors(n_supix*2)
    ver_skel_of_colored = np.zeros([mask.shape[0], mask.shape[1], 3], dtype=np.uint8)

    skeletons_img = -1 * np.ones(shape=mask.shape, dtype=np.int16)
    distances_img = np.zeros(shape=mask.shape, dtype=np.int16)
    skel_of_shifted = -1 * np.ones(shape=mask.shape, dtype=np.int16)
    sp_shifted = -1 * np.ones(shape=mask.shape, dtype=np.int16)

    ###
    #   iterate through frames
    ###
    for i, f in enumerate(onlyfiles):        
        print('[', i, ']. frame is started. Runtime: ', np.round(timeit.default_timer() - start, 2))

        ###
        #   read original and optical flow files
        ###
        img_in = io.imread(join(infolder, f))

        if i < len(onlyfiles)-2:
            of_file = join(offolder, f[:-3]+ 'flo')
        else:
            of_file = join(offolder, onlyfiles[-2][:-3]+ 'flo')
        of_in = computeImg(of_file)

        depth_in = resize(io.imread(join(depthfolder, f[:-3] + 'png')), (img_in.shape[0], img_in.shape[1]), mode='constant', anti_aliasing=True)
    
        ###
        #   calculate boruvka superpixel segmentation
        ###
        bosupix = boruvka_superpixel.BoruvkaSuperpixel()
        label_out = np.zeros((img_in.shape[0], img_in.shape[1], 1), dtype=np.int16)

        if isdir(edgefolder):
            img_edge = io.imread(join(edgefolder, f[:-3] + 'png'))
        else:
            img_edge = np.zeros((img_in.shape[0], img_in.shape[1]), dtype=np.uint8)

        if exists(join(outfolder_skeleton, f[:-4] + '_' + str(mat_thr) + '_seed.npy')) and debug_mode:
            seed_id_out = np.load(join(outfolder_skeleton, f[:-4] + '_' + str(mat_thr) + '_seed.npy')).astype(np.int16)
        elif i == 0: # first frame - RGB
            bosupix.build_2d(img_in, img_edge)
            bosupix.label_o(n_supix, label_out)
            seed_id_out = np.copy(label_out)

            if debug_mode:
                np.save(join(outfolder_skeleton, f[:-4] + '_' + str(mat_thr) + '_seed'), seed_id_out)
        else: # other frames - seeds
            bosupix.build_2d(img_in, img_edge, skel_thr_img)
            bosupix.label(0)
            seed_id_out = bosupix.seed_id()

            if debug_mode:
                np.save(join(outfolder_skeleton, f[:-4] + '_' + str(mat_thr) + '_seed'), seed_id_out)

        
        if seed_id_out.ndim == 3:
            seed_id_out = seed_id_out[:, :, 0]

        print('[', i, ']. Boruvka is ready. Runtime: ', np.round(timeit.default_timer() - start, 2))

        ###
        #   shift superpixels with optical flow
        ###
        of_avg = average(of_in, seed_id_out)
        depth_avg = average(depth_in, seed_id_out)

        seed_depth_map = []

        for seed_id in np.unique(seed_id_out):
            w_x, w_y = np.where(seed_id_out == seed_id)
            depth = depth_avg[w_x[0], w_y[0]]
            seed_depth_map.append((seed_id, depth))

        seed_depth_map = sorted(seed_depth_map, key=lambda x: x[1], reverse=False)

        ver_skel_of_colored[:, :] = [0, 0, 0]
        skel_of_shifted[:, :] = -1
        sp_shifted[:, :] = -1
        for seed_id, _ in seed_depth_map:

            # select region
            w_sp_x, w_sp_y = np.where(seed_id_out == seed_id)
            
            # get mean optical flow of the selected superpixel
            of = of_avg[w_sp_x[0], w_sp_y[0]]
            
            sp_x = w_sp_x + int(of[1])
            sp_y = w_sp_y + int(of[0])

            sp_x[sp_x < 0] = 0
            sp_y[sp_y < 0] = 0
            sp_x[sp_x >= max_height] = max_height-1
            sp_y[sp_y >= max_width] = max_width-1


            sp_shifted[sp_x, sp_y] = seed_id  # skel_of_shifted[sp_x, sp_y]

        print('[', i, ']. calculate new positions with optical flow. Runtime: ', np.round(timeit.default_timer() - start, 1))
        
        # select sps under the gt mask
        if i == 0 and sp_mask_ratio != -1:
            
            assert (sp_mask_ratio >= 0 and sp_mask_ratio <= 1), "Argument sp_mask_ratio must be between 0 and 1."

            mask_skel_ids = np.unique(seed_id_out[mask > 0])
            filtered_mask_skel_ids = []
            for mask_skel_id in mask_skel_ids:
                img_area = np.sum(seed_id_out == mask_skel_id)
                mask_area = np.sum(np.logical_and(seed_id_out == mask_skel_id, mask > 0))

                # filter skeletons based on the sp-mask ratio
                if float(mask_area) / float(img_area) > sp_mask_ratio:
                    filtered_mask_skel_ids.append(mask_skel_id)

            mask_skel_ids = np.copy(filtered_mask_skel_ids)

        elif i == 0:

            mask_skel_ids = np.unique(seed_id_out[mask > 0])
            mask_skel_ids = mask_skel_ids[1:] # remove first element (-1: background id)


        ###
        #   calculate medial axis transform for every superpixel of the frame and save
        #   and False: dont use the npy files, calculate them always
        ###
        if exists(join(outfolder_skeleton, f[:-4] + '_skel.npy')) and exists(join(outfolder_skeleton, f[:-4] + '_dskel.npy')) and debug_mode and False:
            skeletons_img = np.load(join(outfolder_skeleton, f[:-4] + '_skel.npy')).astype(np.int16)
            distances_img = np.load(join(outfolder_skeleton, f[:-4] + '_dskel.npy')).astype(np.int16)
            print('[', i, ']. skeleton is loaded. Runtime: ', np.round(timeit.default_timer() - start, 2))

        else:
            skeletons_img[:, :] = -1
            distances_img[:, :] = 0

            for seed_id in np.unique(sp_shifted):
                if seed_id == -1: continue

                # foreground superpixel: calculate skeleton
                if seed_id in mask_skel_ids:
                    props_mask = np.zeros(shape=sp_shifted.shape, dtype=np.uint8)
                    props_mask[sp_shifted == seed_id] = 255
                    
                    if int(technique) == 0:
                        # foreground superpixel: calculate skeleton or thinning
                        skel, dist = calculate_skeleton(props_mask, is_binary=False)

                        # weigth skeleton with the distance, then scale (normalize) between [0, 255]
                        dskel = dist * skel
                        dskel_ip = np.interp(dskel, (dskel.min(), dskel.max()), (0, 255)).astype(np.int16)
                    else:
                        skel = calculate_thinning(props_mask)
                        dskel_ip = skel
                    
                    # calculate merged skeletons and distances
                    skeletons_img[skel] = seed_id
                    distances_img += dskel_ip
                else:
                    # background superpixel: only centroid
                    cx, cy = np.mean(np.nonzero(sp_shifted == seed_id), axis=1)
                    skeletons_img[int(cx), int(cy)] = seed_id
                    distances_img[int(cx), int(cy)] = 255
 
            distances_img[distances_img == 0] = -1

            if debug_mode:
                np.save(join(outfolder_skeleton, f[:-4] + '_skel'), skeletons_img)
                np.save(join(outfolder_skeleton, f[:-4] + '_dskel'), distances_img)

            print('[', i, ']. skeleton is calculated. Runtime: ', np.round(timeit.default_timer() - start, 2))


        ###
        #   pruning - lower than threshold is background
        ###
        skel_thr_img = np.copy(skeletons_img)
        skel_thr_img[distances_img < mat_thr] = -1

        if i == 0:
            # set outflown skeletons to background using the gt mask
            for mask_skel_id in mask_skel_ids:
                skel_x, skel_y = np.where(skel_thr_img == mask_skel_id)
                for skel_ind in range(len(skel_x)):
                    if mask[skel_x[skel_ind], skel_y[skel_ind]] == 0:
                        skel_thr_img[skel_x[skel_ind], skel_y[skel_ind]] = -1 
        
        # draw mask
        mask[:, :] = 0
        for mask_skel_id in mask_skel_ids:
            mask[seed_id_out==mask_skel_id] = 255

        print('[', i, ']. save mask. Runtime: ', np.round(timeit.default_timer() - start, 1))
        skimage.io.imsave(join(outfolder_segmentation, f[:-4] + '.png'), mask)

        print('[', i, ']. visualize. Runtime: ', np.round(timeit.default_timer() - start, 1))
        visualize_skeletons(join(outfolder_test, 'skel_' + f[:-4] + '.png'), max_height, max_width, n_supix, colors,
                    img_in, seed_id_out, sp_shifted, skeletons_img)

        visualize_skeletons(join(outfolder_test, 'thr_' + f[:-4] + '.png'), max_height, max_width, n_supix, colors,
                    img_in, seed_id_out, sp_shifted, skel_thr_img)


        print('[', i, ']. frame is ready. Runtime: ', np.round(timeit.default_timer() - start, 1))

        if debug_mode and i>4:
            print("[Debug]: exit")
            sys.exit()
 
    print(outfolder)
    try:
        cv2.destroyAllWindows()
    except:
        pass


# join(outfolder_segmentation, f[:-4] + '.png')
def visualize_skeletons(fn, max_height, max_width, n_supix, colors, img_in, seed_id_out, input, mask):
    output = np.zeros(shape=(max_height, max_width, 3), dtype=np.uint8)
    img_color = average(img_in, seed_id_out)

    for seed_id in np.unique(input):

        if seed_id == -1:
            output[input == seed_id] = [0, 0, 0]
        else:
            cx, cy = np.where(input == seed_id)        
            output[input == seed_id] =  img_color[cx, cy][0]

        output[mask > 0] = [255, 255, 0]
    
    skimage.io.imsave(fn, output)


def process_folders(infolder, maskfolder, offolder, edgefolder, depthfolder, outfolder, 
                    technique, sp_mask_ratio, n_supix, mat_thr):
    
    if not exists(outfolder):
        makedirs(outfolder)

    scene_ids = listdir(infolder)
    scene_ids.sort()

    if debug_mode:
        scene_ids = [scene_ids[1]]
    
    
    params = []
    for scene_id in scene_ids:
        
        params.append((join(infolder, scene_id), 
                       join(maskfolder, scene_id), 
                       join(offolder, scene_id), 
                       join(edgefolder, scene_id),
                       join(depthfolder, scene_id),
                       join(outfolder, scene_id), 
                       technique, 
                       sp_mask_ratio,
                       n_supix, 
                       mat_thr))

    if debug_mode:
        boruvkasupix2D(params[0])
    else:
        pool = Pool(processes=2) 
        pool.map(boruvkasupix2D, params)


def parse_arguments(argv):
    description = ('calculate superpixels, '
            'output orig image folder with color averaged within superpixels')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('infolder',
            help='input image folder')
    parser.add_argument('maskfolder',
            help='mask image folder')
    parser.add_argument('offolder',
            help='OF folder')
    parser.add_argument('edgefolder',
            help='edge folder')
    parser.add_argument('depthfolder',
            help='depth folder')
    parser.add_argument('outfolder',
            help='output image folder')
    parser.add_argument('technique',
            help='0 - skeletonize, 1 - thinning')
    parser.add_argument('sp_mask_ratio',
            type=float,
            help='sp flow tolerance over mask border')
    parser.add_argument('n_supix',
            type=int,
            help='number of superpixels')
    parser.add_argument('mat_thr',
            type=int,
            help='medial axis transform distance threshold')
    args = parser.parse_args(argv)

    return args


def main():
    '''Video propagation with Boruvka superpixels and Medial Axis Transform.

    Example:
        cd videopropagation
        python3 boruvka_mat.py
            /path/to/DAVIS/JPEGImages/480p      # infolder
            /path/to/DAVIS/Annotations/480p     # maskfolder
            /path/to/pwc_out/DAVIS              # offolder
            /path/to/hed_out                    # edgefolder
            /path/to/results/davis              # outfolder
            1
            0.5                                 # sp_mask_ratio
            64                                  # n_supix
            64                                  # mat_thr
    '''
    args = parse_arguments(sys.argv[1:])
    process_folders(**args.__dict__)


if __name__ == '__main__':
    sys.exit(main())

