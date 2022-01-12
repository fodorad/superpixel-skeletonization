#!/usr/bin/env python3
# pylint: disable=E1101
# pylint: disable=E0401
# pylint: disable=E0611

# standard libs
import os
import sys
import warnings
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

# 3rd party libs
from skimage.io import imsave

# local files
import dataset.load_data as load_data
from algorithms.baseline import baseline, baseline2
from algorithms.single_skeleton import single_skeleton


def call_method(params) -> None:
    """
    This method...
        loads the data
        calls the desired method
        saves the output masks

    Args:
        params (Tuple): contains the following:
            video (VideoLoader): video file loader
            out_folder (Path): output image folder
            input_id (int): input feature configuration id
            n_supix (int): number of superpixels
            overlap (float): accepted overlap of superpixels
            disk_thr (float): disk threshold
            skeleton_thr (float): skeleton threshold
            color_space (str): color space of input images
            depth_prefactor (float): prefactor of depth information

    """
    video, method, out_folder, input_id, n_supix, overlap, disk_thr, skeleton_thr, \
    color_space, depth_prefactor = params

    # load data
    video.load(color_space=color_space, depth_prefactor=depth_prefactor)

    # run method
    if method == 'single-skeleton':
        masks = single_skeleton(video.img_ins,
                                video.gt_ins,
                                video.of_ins,
                                video.depth_ins,
                                video.edge_ins,
                                input_id,
                                n_supix,
                                overlap,
                                disk_thr,
                                skeleton_thr)

    elif method == 'baseline':
        masks = baseline(video.img_ins,
                          video.gt_ins,
                          video.of_ins,
                          video.edge_ins,
                          n_supix)

    elif method == 'baseline2':
        masks = baseline2(video.img_ins,
                          video.gt_ins,
                          video.of_ins,
                          video.edge_ins,
                          n_supix,
                          overlap)
    else:
        raise NotImplementedError()

    # save masks
    out_folder.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, f in enumerate(video.in_files):
            imsave(str(out_folder / (f[:-4] + '.png')), masks[:,:,i])


def parse_arguments(argv):
    """Parse and process command line arguments

    Args:
        argv: input arguments

    Returns:
        parsed command line arguments for each video

    """
    description = ('Runner script for baseline2 and single-skeleton methods, '
                   'Optical flow based superpixel and mask skeleton propagation')
    parser = argparse.ArgumentParser(description=description)
    lambda_path = lambda p: Path(p).absolute()

    # required
    parser.add_argument('method',
                        type=str,
                        help='name of the algorithm \
                              [baseline, baseline2, single-skeleton]')
    parser.add_argument('outfolder',
                        type=lambda_path,
                        help='output image folder')

    # optional
    parser.add_argument('--db',
                        type=str,
                        default='davis',
                        help='name of the database')
    parser.add_argument('--in_folder',
                        type=str,
                        default=None,
                        help='input image folder')
    parser.add_argument('--mask_folder',
                        type=str,
                        default=None,
                        help='mask image folder')
    parser.add_argument('--of_folder',
                        type=str,
                        default=None,
                        help='forward OF folder')
    parser.add_argument('--edge_folder',
                        type=str,
                        default=None,
                        help='edge folder')
    parser.add_argument('--depth_folder',
                        type=str,
                        default=None,
                        help='depth folder')
    parser.add_argument('--input_id',
                        type=int,
                        default=0,
                        help='input feature configuration id \
                              0: color + hed + depth, 1: hed, 2: color, \
                              3: color + hed, 4: depth, 5: depth + hed, \
                              6: of, 7: of + hed')
    parser.add_argument('--n_supix',
                        type=int,
                        default=512,
                        help='number of superpixels')
    parser.add_argument('--overlap',
                        type=float,
                        default=0.5,
                        help='accepted overlap of superpixels')
    parser.add_argument('--disk_thr',
                        type=int,
                        default=0,
                        help='disk threshold applied to the draft mask')
    parser.add_argument('--skeleton_thr',
                        type=float,
                        default=0,
                        help='skeleton threshold')
    parser.add_argument('--color_space',
                        type=str,
                        default='lab',
                        help='input image color space [rgb, lab]')
    parser.add_argument('--depth_prefactor',
                        type=float,
                        default=0.5,
                        help='prefactor of the depth information')
    parser.add_argument('--nproc',
                        type=int,
                        default=max(cpu_count()-2, 1),
                        help='number of processes')
    parser.add_argument('--d',
                        action='store_true',
                        help='test on 1 video')

    args = parser.parse_args(argv)
    return args


def prepare_dataset(args):

    assert args.method in ['baseline', 'baseline2', 'single-skeleton'], 'Invalid method.'
    assert args.db in ['davis'], 'Invalid database. The only supported one is [davis].'
    assert args.input_id in range(8), 'Input id must be between 0 and 7.'
    assert args.n_supix >= 0, 'Number of superpixels should be greater than 0.'
    assert args.disk_thr >= 0, 'Threshold should be greater than 0.'
    assert args.skeleton_thr >= 0, 'Threshold should be greater than 0.'
    assert args.overlap >= 0 and args.overlap <= 1, 'Overlap should be between 0 and 1.'
    assert args.depth_prefactor >= 0, 'Depth prefactor should be greater than or equal to 0.'
    assert args.color_space in ['rgb', 'lab'], 'Invalid color space. Choose one from [rgb, lab].'

    params = []
    scene_ids = args.in_folder if args.in_folder is not None else os.listdir(load_data.DAVIS_IN_DIR)
    scene_ids.sort()

    if args.d:
        scene_ids = [scene_ids[10]] # car-shadow

    for scene_id in scene_ids:
        video = load_data.DavisVideo(scene_id=scene_id,
                                     in_folder=args.in_folder,
                                     mask_folder=args.mask_folder,
                                     of_folder=args.of_folder,
                                     edge_folder=args.edge_folder,
                                     depth_folder=args.depth_folder)
        params.append((video,
                       args.method,
                       args.outfolder / scene_id,
                       args.input_id,
                       args.n_supix,
                       args.overlap,
                       args.disk_thr,
                       args.skeleton_thr,
                       args.color_space,
                       args.depth_prefactor))

    return params


def main() -> None:
    args = parse_arguments(sys.argv[1:])
    params = prepare_dataset(args)
    pool = Pool(processes=args.nproc)
    for _ in pool.imap_unordered(call_method, params):
        pass


if __name__ == '__main__':
    sys.exit(main())


