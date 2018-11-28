# --------------------------------------------------------
# Fast R-CNN mobility aids version
# Licensed under The MIT License [see LICENSE for details]
# Written by Andres Vasquez, Marina Kollmitz
# --------------------------------------------------------

import datasets
import datasets.mobility_aids
import os
import datasets.imdb
import numpy as np
import utils.cython_bbox
import cPickle
import yaml
import pandas as pd

class mobility_aids(datasets.imdb):
    def __init__(self, image_set, roidb_txtfiles_path, imageset_file, image_folder, annotations_folder):
        datasets.imdb.__init__(self, image_set)
        
        self._image_set = image_set
        self._roidb_txtfiles_path = roidb_txtfiles_path
        self._image_folder = image_folder
        self._annotations_folder = annotations_folder
        self._imageset_file = imageset_file
        
        self._classes = ('__background__', # always index 0
                         'pedestrian','wheelchair','push_wheelchair',
                         'crutches','walking_frame')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        
        self._image_ext = ['.png']
        self._image_index = self._load_image_set_index()

        # Default to roidb handler
        self._roidb_handler = self.txtfiles_roidb

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._image_folder, index + ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        assert os.path.exists(self._imageset_file), \
                'Path does not exist: {}'.format(self._imageset_file)
        with open(self._imageset_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        
        
        gt_roidb = [self._load_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def txtfiles_roidb(self):
        """
        return the database of regions of interest from textfile format:
        xmin ymin xmax ymax
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_txtfiles_roidb.pkl'.format(self.name))

        gt_roidb = self.gt_roidb()
        # Remove frames with no object annotations
        orig_image_index = self.image_index
        removeset = set()

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} textfiles roidb loaded from {}'.format(self.name, cache_file)
            if "train" in self._image_set:
                for index in xrange(len(gt_roidb)):
                    if gt_roidb[index] is None:
                        removeset.add(index)
                mod_image_index =  [v for i, v in enumerate(orig_image_index) if i not in removeset]
                del self.image_index[:]
                for index in mod_image_index:
                    self.image_index.append(index)
            return roidb

        if "train" in self._image_set:
            ss_roidb = self._load_txtfiles_roidb(gt_roidb)
            for index in xrange(len(gt_roidb)):
                if gt_roidb[index] is None:
                    removeset.add(index)
            mod_image_index =  [v for i, v in enumerate(orig_image_index) if i not in removeset]
            del self.image_index[:]
            for index in mod_image_index:
                self.image_index.append(index)
            print "Ignoring frames with no object annotations"
            print "len gt roidb", len(gt_roidb), "len ss roidb", len(ss_roidb), "len image index", len(self.image_index)
            gt_roidb = [index for index in gt_roidb if index is not None]
            ss_roidb = [index for index in ss_roidb if index is not None]
            print "len gt roidb", len(gt_roidb), "len ss roidb", len(ss_roidb), "len image index", len(self.image_index)
            assert len(gt_roidb) == len(ss_roidb) == len(self.image_index)

            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else: 
            roidb = self._load_txtfiles_roidb(None)
            "len roidb", len(roidb), "len image index", len(self.image_index)
            assert len(roidb) == len(self.image_index)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote textfiles roidb to {}'.format(cache_file)

        return roidb

    def _load_txtfiles_roidb(self, gt_roidb):
        assert os.path.exists(self._roidb_txtfiles_path), \
               'roidb textfile not found at: {}'.format(self._roidb_txtfiles_path)

        box_list = []
        depth_list = []

        for i in xrange(self.num_images):
            filename = os.path.join(self._roidb_txtfiles_path, self.image_index[i] + '.txt')
            print("filename: ", filename)
            raw_data = pd.read_csv(filename, delimiter=" ").as_matrix()
            
            box_list.append((raw_data[:,0:4]).astype(np.uint16))
            
            if raw_data.shape[1] > 4:
                depth_list.append(raw_data[:,-1])
            else:
                depth_list.append(-1*np.ones([raw_data.shape[0], 1]))

        return self.create_roidb_from_box_list(box_list, gt_roidb, depth_list)

    def _load_annotation(self, index):
        """
        Load image and bounding boxes info from yaml annotation files.
        """
        
        assert os.path.exists(self._annotations_folder), \
                'Path does not exist: {}'.format(self._annotations_folder)
        
        boxes = []
        gt_classes = []
        overlaps = []
        
        annotation_file = os.path.join(self._annotations_folder, index + '.yml')
        with open(annotation_file,'r') as fid:
            #hack to skip the ehader and read yml correctly
            #fix yaml parsing issue
            filedata = fid.read()
            filedata = filedata.replace("YAML:1.0", "YAML 1.0\n---") 

            #load yaml file
            data_loaded = yaml.load(filedata)
            
            #check if there are labeled instances in the file
            if 'object' in data_loaded['annotation']:
                #parse each object instance
                for inst in data_loaded['annotation']['object']:
                    class_name = inst['name']
                    class_id = self._class_to_ind[class_name]
                    bbox = inst['bndbox']
                    xmin = int(bbox['xmin'])
                    ymin = int(bbox['ymin'])
                    xmax = int(bbox['xmax'])
                    ymax = int(bbox['ymax'])
                    boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                    gt_classes.append(class_id)
                    overlaps = np.zeros(len(self.classes))
                    overlaps[class_id] = 1.0
                
                return {'boxes' : np.array(boxes),
                        'gt_classes': np.array(gt_classes),
                        'gt_overlaps' : np.array(overlaps),
                        'flipped' : False}
            else:
                print "empty yml ", annotation_file
                return None

    def _write_results_file(self, all_boxes, output_dir):
        # VOCdevkit/results/comp4-44503_det_test_aeroplane.txt
        #path = os.path.join(self._devkit_path, 'results', self.name, comp_id + '_')
        path = output_dir
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            #print 'Writing {} results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            print 'Writing results file ', filename
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        if(k == 0):
                            f.write('{:s} {:d} {:d} {:d} {:d} {:.6f};'.
                                format(index, int(dets[k, 0]) + 1, int(dets[k, 1]) + 1,
                                        int(dets[k, 2]) + 1, int(dets[k, 3]) + 1, float(dets[k, 4])))
                        else:
                            f.write(' {:d} {:d} {:d} {:d} {:.6f};'.
                                format( int(dets[k, 0]) + 1, int(dets[k, 1]) + 1,
                                        int(dets[k, 2]) + 1, int(dets[k, 3]) + 1, float(dets[k, 4]))) 
                    f.write('\n')        

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_results_file(all_boxes, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    dataset_root = '/home/kollmitz/datasets/mobility-aids/'
    roidb_textfiles_path = dataset_root + 'roidb_segmentation/'
    imageset_file = dataset_root + 'ImageSets/TrainSet_RGB.txt'
    image_folder = dataset_root + 'Images'
    annotations_folder = dataset_root + 'Annotations_RGB'
    d = mobility_aids('RGB_train', roidb_textfiles_path, imageset_file, image_folder, annotations_folder)
    res = d.roidb
    print "done"
    from IPython import embed; embed()
