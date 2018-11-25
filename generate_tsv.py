#!/usr/bin/env python


"""Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a 
   separate tsv file that can be merged later (e.g. by using merge_tsv function). 
   Modify the load_image_ids script as necessary for your data location. """


# Example:
# ./tools/generate_tsv.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --out test2014_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split coco_test2014


#import sys
#sys.path.insert(0, './../bottom-up-attention/lib/')
#sys.path.insert(0, './../fast-rcnn/lib/')

from tqdm import tqdm

import os
print os.getcwd()
os.chdir('../bottom-up/')
print os.getcwd()

import sys
sys.path.insert(0, './caffe/python/')
sys.path.insert(0, './lib/')
sys.path.insert(0, './tools/')


print("A")
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

print("B")



import caffe
import argparse
import pprint
import time, os, sys
import base64
import numpy as np
import cv2
import csv
from multiprocessing import Process
import random
import json

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36. 
MIN_BOXES = 10
MAX_BOXES = 100

def load_image_ids(folder):
    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
    split = []
    for file in os.listdir(folder):
        if file.endswith('.jpg'):
            _id = file.split('.')[0]
            split.append((folder+file,_id))      
    return split

    
def get_detections_from_im(net, im_file, image_id, conf_thresh=0.2):

    im = cv2.imread(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
   
    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes' : len(keep_boxes),
        'boxes': json.dumps(cls_boxes[keep_boxes].tolist()),
        'features': json.dumps(pool5[keep_boxes].tolist())
    }   

def generate_tsv(prototxt, weights, image_ids, outfile):
    # First check if file exists, and if it is complete

    caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffe.TEST, weights=weights)
    with open(outfile, 'ab') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)   
        for im_file,image_id in tqdm(image_ids):
            writer.writerow(get_detections_from_im(net, im_file, image_id))


if __name__ == '__main__':



    base_p = '../vqa-belen/'
    cfg_file = base_p+'faster_rcnn_end2end_resnet.yml'
    prototxt = base_p+'test.prototxt'
    caffemodel = base_p+'resnet101_faster_rcnn_final.caffemodel'
    data_path = base_p+'data/'
    outfile = base_p+'outputs/features.tsv'

    cfg_from_file(cfg_file)

    #print('Using config:')
    #pprint.pprint(cfg_file)
    assert cfg.TEST.HAS_RPN

    image_ids = load_image_ids(data_path)
    random.seed(10)
    random.shuffle(image_ids)

    # Split image ids between gpus
    caffe.init_log()
    
    generate_tsv(prototxt, caffemodel, image_ids, outfile)        
             
    print("DONE!!")

