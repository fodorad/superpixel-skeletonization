#!/bin/sh

# This script downloads the outputs of HED, MegaDepth and PWC-Net on DAVIS and unzips it.
#
# The outputs were created with the following implementations:
# HED: https://github.com/sniklaus/pytorch-hed
# MegaDepth: https://github.com/lixx2938/MegaDepth
# PWC-Net: https://github.com/sniklaus/pytorch-pwc
#
# Install megatools on Debian/Ubuntu:
# sudo apt-get install megatools
#
# Note:
#   Files might not be available right now.
#   Write a mail to foauaai@inf.elte.hu

DIR=$(pwd)/'db'
HED_FILE='hed_out.zip'
D_FILE='depth_out.zip'
OF_FILE='pwc_out.zip'

echo 'Download HED outputs... (416 MB)'
megadl 'https://mega.nz/#!HBhkzCjB!ZPasYJRoO36oyyRATT6ePhHPfDuVOS8iBqwU1ahfcDI' ./$HED_FILE

echo 'Download MegaDepth outputs... (835 MB)'
megadl 'https://mega.nz/#!zEZVkY7T!al34K6VrAbleqtblz8mngZieGNWPqlQBD4OWkLYbzlE' ./$D_FILE

echo 'Download PWC-Net outputs... (9.5 GB)'
megadl 'https://mega.nz/#!HZZVCaTR!ETtkKaFBX4GltWH_JtS5Zb19ZG9wyVjfxJ3Pr7gqzug' ./$OF_FILE

echo 'Unzipping holistic edges...'
unzip $HED_FILE -d $DIR

echo 'Unzipping depth...'
unzip $D_FILE -d $DIR

echo 'Unzipping forward optical flow...'
unzip $OF_FILE -d $DIR

rm -rf $HED_FILE
rm -rf $D_FILE
rm -rf $OF_FILE


