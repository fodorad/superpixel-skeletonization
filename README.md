# superpixel-skeletonization

Code for the paper "Skeletonization Combined with Deep Neural Networks for Superpixel Temporal Propagation", IJCNN 2019

## Installion

Clone repository and install python dependencies.

The code was created on Python 3.8 using the following libraries:

- numpy
- opencv-python
- scikit-image
- cython

```bash
git clone https://github.com/fodorad/superpixel-skeletonization
cd superpixel-skeletonization
pip install --user -r requirements.txt
```

The code uses [Borůvka Superpixel](https://github.com/semiquark1/boruvka-superpixel).
Make sure, that the submodule is properly initialized.

Newer versions of Git will do this automatically, but older versions will require you to explicitly tell Git to download the contents of boruvka-superpixel:

```bash
git submodule update --init --recursive
```

Build python module in boruvka-superpixel.

```bash
cd ./external/boruvka-superpixel
make module && cd ../..
```

You can download the [DAVIS](https://github.com/davisvideochallenge/davis) database using the following script:

```bash
bash ./dataset/get_davis_2016.sh
```

The code uses the outputs of the following Deep Neural Network architectures:

- [MegaDepth](https://github.com/lixx2938/MegaDepth) for depth estimation
- [PWC-Net](https://github.com/sniklaus/pytorch-pwc) for optical flow estimation
- [Holistically-Nested Edge Detection](https://github.com/sniklaus/pytorch-hed) for edge informations.

## Example usage

The main script is ``run.py``. For all the command line arguments check ``python run.py -h``.

```bash
# run baseline2 method
python run.py baseline2 ./results/baseline2_out --n_supix 512 --color_space lab

# run single-skeleton method
python run.py single-skeleton ./results/single-skeleton_out --n_supix 512 --color_space lab
```

## Contact

* Ádám Fodor -- foauaai@inf.elte.hu


## Citation

If you use this software in your research, please cite it as below.


```
Á. Fodor, Á. Fóthi, L. Kopácsi, E. Somfai and A. Lőrincz, "Skeletonization Combined with Deep Neural Networks for Superpixel Temporal Propagation," 2019 International Joint Conference on Neural Networks (IJCNN), 2019, pp. 1-7, doi: 10.1109/IJCNN.2019.8852391.
```

```
@INPROCEEDINGS{8852391,
  author={Fodor, Ádám and Fóthi, Áron and Kopácsi, László and Somfai, Ellák and Lőrincz, András},
  booktitle={2019 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Skeletonization Combined with Deep Neural Networks for Superpixel Temporal Propagation}, 
  year={2019},
  volume={},
  number={},
  pages={1-7},
  doi={10.1109/IJCNN.2019.8852391}}
```

