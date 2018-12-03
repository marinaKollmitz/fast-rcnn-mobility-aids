# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets.pascal_voc
from datasets.mobility_aids import mobility_aids
import numpy as np
import os

def _selective_search_IJCV_top_k(split, year, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = datasets.pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year))

# Set up voc_<year>_<split>_top_<k> using selective search "quality" mode
# but only returning the first k boxes
for top_k in np.arange(1000, 11000, 1000):
    for year in ['2007', '2012']:
        for split in ['train', 'val', 'trainval', 'test']:
            name = 'voc_{}_{}_top_{:d}'.format(year, split, top_k)
            __sets[name] = (lambda split=split, year=year, top_k=top_k:
                    _selective_search_IJCV_top_k(split, year, top_k))

###set up mobilityaids datasets

#train RGB
name = 'mobilityaids_train_RGB'
dataset_path = os.path.dirname(os.path.realpath(__file__)) + '/data/mobility-aids/'
imageSet = name
roidb_txtfiles_path = dataset_path + 'roidb_sliding/'
imageset_file = dataset_path + 'ImageSets/TrainSet_RGB.txt'
image_folder = dataset_path + 'Images/'
annotations_folder = dataset_path + 'Annotations_RGB/'

mobility_aids(imageSet, roidb_txtfiles_path, imageset_file, image_folder, annotations_folder)

__sets[name] = (lambda : mobility_aids(imageSet, roidb_txtfiles_path, imageset_file, image_folder, annotations_folder))

#train DepthJet
name = 'mobilityaids_train_DJ'
dataset_path = os.path.dirname(os.path.realpath(__file__)) + '/data/mobility-aids/'
imageSet = name
roidb_txtfiles_path = dataset_path + 'roidb_sliding/'
imageset_file = dataset_path + 'ImageSets/TrainSet_DepthJet.txt'
image_folder = dataset_path + 'DepthJet/'
annotations_folder = dataset_path + 'Annotations_DepthJet/'

mobility_aids(imageSet, roidb_txtfiles_path, imageset_file, image_folder, annotations_folder)

__sets[name] = (lambda : mobility_aids(imageSet, roidb_txtfiles_path, imageset_file, image_folder, annotations_folder))

#test RGB segmentation
name = 'mobilityaids_test_RGB_segmentation'
dataset_path = os.path.dirname(os.path.realpath(__file__)) + '/data/mobility-aids/'
imageSet = name
roidb_txtfiles_path = dataset_path + 'roidb_segmentation/'
imageset_file = dataset_path + 'ImageSets/TestSet1.txt'
image_folder = dataset_path + 'Images/'
annotations_folder = dataset_path + 'Annotations_RGB/'

#test DepthJet segmentation
name = 'mobilityaids_test_DJ_segmentation'
dataset_path = os.path.dirname(os.path.realpath(__file__)) + '/data/mobility-aids/'
imageSet = name
roidb_txtfiles_path = dataset_path + 'roidb_segmentation/'
imageset_file = dataset_path + 'ImageSets/TestSet1.txt'
image_folder = dataset_path + 'DepthJet/'
annotations_folder = dataset_path + 'Annotations_DepthJet/'

__sets[name] = (lambda : mobility_aids(imageSet, roidb_txtfiles_path, imageset_file, image_folder, annotations_folder))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
