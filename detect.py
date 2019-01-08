from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from person_detect.util import *
import argparse
import os
import os.path as osp
from person_detect.darknet import Darknet
from person_detect.preprocess import prep_image, inp_to_image
import pandas as pd
import random
import pickle as pkl
import itertools


CUDA = torch.cuda.is_available()

class PersonDetect():
    '''行人检测'''

    def __init__(self, weights='person_detect/yolov3.weights',
                 dist='person_detect/imgs_dist', batch_size=1, confidence=0.5,
                 nms_thesh=0.4, scales='1,2,3',
                 reso='416', cfgfile='person_detect/cfg/yolov3.cfg'
                 ):
        self.dist = dist
        self.batch_size = int(batch_size)
        self.num_classes = 80
        self.confidence = float(confidence)
        self.nms_thesh = float(nms_thesh)
        self.scales = scales
        self.classes = load_classes('person_detect/data/coco.names')
        self.model = Darknet(cfgfile)
        self.model.load_weights(weights)
        print("Network successfully loaded")
        self.model.net_info["height"] = reso
        self.inp_dim = int(self.model.net_info["height"])
        if CUDA:
            self.model.cuda()
        self.model.eval()
        if not os.path.exists(self.dist):
            os.makedirs(self.dist)

    def detect(self, imlist):
        '''
        args:
            imlist(List): list of images path
        '''
        batches = list(
            map(prep_image, imlist, [self.inp_dim for x in range(len(imlist))]))
        im_batches = [x[0] for x in batches]
        orig_ims = [x[1] for x in batches]
        im_dim_list = [x[2] for x in batches]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

        if CUDA:
            im_dim_list = im_dim_list.cuda()

        leftover = 0

        if (len(im_dim_list) % self.batch_size):
            leftover = 1
        if self.batch_size != 1:
            num_batches = len(imlist) // self.batch_size + leftover
            im_batches = [torch.cat((im_batches[i*self.batch_size: min((i + 1)*self.batch_size,
                                                                       len(im_batches))])) for i in range(num_batches)]
        i = 0
        write = False
        for batch in im_batches:
            if CUDA:
                batch = batch.cuda()

            with torch.no_grad():
                prediction = self.model(Variable(batch), CUDA)
            prediction = write_results(
                prediction, self.confidence,
                self.num_classes, nms=True,
                nms_conf=self.nms_thesh
            )

            if type(prediction) == int:
                i += 1
                continue

            prediction[:, 0] += i*self.batch_size

            if not write:
                output = prediction
                write = 1
            else:
                output = torch.cat((output, prediction))
            i += 1

            if CUDA:
                torch.cuda.synchronize()

        try:
            output
        except NameError:
            print("No detections were made")
            return

        im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

        scaling_factor = torch.min(self.inp_dim/im_dim_list, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (self.inp_dim - scaling_factor *
                              im_dim_list[:, 0].view(-1, 1))/2
        output[:, [2, 4]] -= (self.inp_dim - scaling_factor *
                              im_dim_list[:, 1].view(-1, 1))/2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(
                output[i, [1, 3]], 0.0, im_dim_list[i, 0])
            output[i, [2, 4]] = torch.clamp(
                output[i, [2, 4]], 0.0, im_dim_list[i, 1])

        colors = pkl.load(open("person_detect/pallete", "rb"))

        det_names = []
        output_imgs = []
        i = 0
        for x in output:
            c1 = tuple(x[1:3].int())
            c2 = tuple(x[3:5].int())

            img = orig_ims[int(x[0])]
            cls = int(x[-1])
            label = "{0}".format(self.classes[cls])
            if label != 'person':
                continue
            output_imgs.append(img[c1[1]:c2[1], c1[0]:c2[0]])
            det_names.append("{}/det_{}_{}".format(self.dist,
                                                   i, imlist[0].split("/")[-1]))
            i += 1

        list(map(cv2.imwrite, pd.Series(det_names), output_imgs))

        torch.cuda.empty_cache()
        print(det_names)
        return det_names


if __name__ == '__main__':
    # python detect.py --images imgs --det det --weight yolov3.weights
    detect = PersonDetect()
    imlist = ['/Users/xuxp/Desktop/zhoudongyu.png']
    detect.detect(imlist)
