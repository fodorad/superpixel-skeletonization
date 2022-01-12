#!/usr/bin/env python3
# pylint: disable=E0401

# standard libs
import os
from pathlib import Path
from abc import ABC, abstractmethod

# 3rd party libs
import numpy as np
from skimage import color
from skimage.io import imread
from skimage.transform import resize


# Davis database default constants
DB_DIR = Path(__file__).resolve().parents[1] / 'db'
DAVIS_DIR = DB_DIR / 'DAVIS'
DAVIS_IN_DIR = DAVIS_DIR / 'JPEGImages' / '480p'
DAVIS_GT_DIR = DAVIS_DIR / 'Annotations' / '480p'
DAVIS_OF_DIR = DB_DIR / 'pwc_out'
DAVIS_OFR_DIR = DB_DIR / 'pwc_r_out'
DAVIS_HED_DIR = DB_DIR / 'hed_out'
DAVIS_D_DIR = DB_DIR / 'depth_out'


class VideoLoader(ABC):
    """Abstract class for loading a single benchmark video

    Abstract method:
        load: read corresponding input files

    """
    def __init__(self,
                 in_folder: Path,
                 mask_folder: Path,
                 of_folder: Path,
                 edge_folder: Path,
                 depth_folder: Path):

        for folder in [in_folder, mask_folder, of_folder, edge_folder, depth_folder]:
            if folder is not None and not folder.is_dir():
                raise NotADirectoryError('Invalid directory: {}'.format(str(folder)))

        self.in_folder = in_folder
        self.mask_folder = mask_folder
        self.of_folder = of_folder
        self.edge_folder = edge_folder
        self.depth_folder = depth_folder

        # fill after calling concrete subclass constructor
        self.height = None
        self.width = None
        self.length = None

        # fill after calling load
        self.in_files = None
        self.img_ins = None
        self.gt_ins = None
        self.of_ins = None
        self.depth_ins = None
        self.edge_ins = None


    @abstractmethod
    def load(self):
        return


class DavisVideo(VideoLoader):
    """Concrete class for loading a single DAVIS benchmark video 

    """
    def __init__(self,
                 scene_id: str,
                 in_folder: str = None,
                 mask_folder: str = None,
                 of_folder: str = None,
                 edge_folder: str = None,
                 depth_folder: str = None) -> None:
        """Constructor

        Args:
            scene_id (str): Name of the video as well as name of the directory with the input images
            in_folder (str): path of JPEGImages
            mask_folder (str): path of Annotations
            of_folder (str): path of optical flow files *
            edge_folder (str): path of holistic edge files *
            depth_folder (str): path of depth files *

        Note:
            * directory tree is supposedly similar to DAVIS

        """
        in_folder = Path(in_folder) / scene_id if in_folder is not None else DAVIS_IN_DIR / scene_id
        mask_folder = Path(mask_folder) / scene_id if mask_folder is not None else DAVIS_GT_DIR / scene_id 
        of_folder = Path(of_folder) / scene_id if of_folder is not None else DAVIS_OF_DIR / scene_id 
        edge_folder = Path(edge_folder) / scene_id if edge_folder is not None else DAVIS_HED_DIR / scene_id 
        depth_folder = Path(depth_folder) / scene_id if depth_folder is not None else DAVIS_D_DIR / scene_id 

        super().__init__(in_folder=in_folder, mask_folder=mask_folder, of_folder=of_folder,
                         edge_folder=edge_folder, depth_folder=depth_folder)

        in_files = [f for f in os.listdir(self.in_folder) if (self.in_folder / f).is_file()]
        in_files.sort()
        assert len(in_files) > 0, 'Missing input images.'
        self.in_files = in_files

        img = imread(str(in_folder / in_files[0]))
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.length = len(in_files)


    def load(self, depth_prefactor: float = 0.5, color_space: str = 'rgb') -> None:
        """Load and prepare files

        Args:
            depth_prefactor (float): prefactor of depth values (Default: 0.5)
            color_space (str): color space of input images. RGB and Lab is supported. (Default: 'rgb') 

        """
        assert color_space in ['rgb', 'lab'], 'Invalid color space. Choose one from [rgb, lab].'

        if color_space == 'rgb':
            self.img_ins = np.zeros(shape=(self.height, self.width, self.length, 3), dtype=np.uint8)

        else: # 'lab'
            self.img_ins = np.zeros(shape=(self.height, self.width, self.length, 3), dtype=np.int16)

        for i, f in enumerate(self.in_files):
            img_in = imread(str(self.in_folder / f)) # rgb

            if color_space == 'lab':
                img_in = color.rgb2lab(img_in).astype(np.int16)

            self.img_ins[:,:,i,:] = img_in

        if self.mask_folder is not None:
            self.gt_ins = np.zeros(shape=(self.height, self.width, self.length), dtype=np.uint8)
            for i, f in enumerate(self.in_files):
                gt_in = imread(str(self.mask_folder / (f[:-4] + '.png')))
                # height x width images are expected
                # in case of an unexpected 3-channel gt mask loading with skimage 
                # (like bear 78)
                if gt_in.ndim == 3:
                    gt_in = gt_in[:,:,0]
                self.gt_ins[:,:,i] = gt_in

        if self.edge_folder is not None:
            self.edge_ins = np.zeros(shape=(self.height, self.width, self.length), dtype=np.int16)
            for i, f in enumerate(self.in_files):
                self.edge_ins[:,:,i] = imread(str(self.edge_folder / (f[:-4] + '.png')))

        if self.depth_folder is not None:
            self.depth_ins = np.zeros(shape=(self.height, self.width, self.length), dtype=np.float)
            for i, f in enumerate(self.in_files):
                self.depth_ins[:,:,i] = depth_prefactor * resize(imread(str(self.depth_folder / (f[:-4] + '.png'))), 
                                                                 (self.height, self.width), 
                                                                 mode='constant', 
                                                                 anti_aliasing=True)

        if self.of_folder is not None:
            self.of_ins = np.zeros(shape=(self.height, self.width, self.length, 2), dtype=np.float)
            for i, f in enumerate(self.in_files):
                # OF (t -> t+1) and OFR (t+1 -> t)
                if i < self.length-1:
                    of_file = self.of_folder / (f[:-4] + '.flo')
                    of = read_flo(str(of_file)) # np.rint(read(str(of_file)))

                    # last frame is black (t -> t+1)
                    # switch row, column ordering
                    self.of_ins[:,:,i,0] = of[:,:,1]
                    self.of_ins[:,:,i,1] = of[:,:,0]


def read_flo(file: str) -> np.ndarray:
    """Read optical flow file .flo

        According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein  
        Contact: dqsun@cs.brown.edu
        Contact: schar@middlebury.edu

        Author: Johannes Oswald, Technical University Munich
        Contact: johannes.oswald@tum.de
        Date: 26/04/2

        For more information, check http://vision.middlebury.edu/flow/

    Args:
        file (str): input file path
    
    Returns:
        (np.ndarray): optical flow as a 2D numpy array
    """
    TAG_FLOAT = 202021.25

    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]

    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]

    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number

    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)

    data = np.fromfile(f, np.float32, count=2*w[0]*h[0])

    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))	
    f.close()

    return flow


