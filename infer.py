from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import glob
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

from args import argument_parser, image_dataset_kwargs, optimizer_kwargs
from torchreid.data_manager import ImageDataManager
from torchreid import models
from torchreid.utils.iotools import check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.torchtools import count_num_param, open_all_layers, open_specified_layers
from torchreid.utils.reidtools import save_ranked_results
from torchreid.eval_metrics import evaluate
from torchreid.optimizers import init_optimizer
from torchreid.transforms import build_transforms
from torchreid.dataset_loader import read_image
from torch.utils.data import Dataset, DataLoader

import pickle

class Bunch(dict):
    def __init__(self, **kwargs):
        super(Bunch, self).__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

# global variables
# parser = argument_parser()
# args = parser.parse_args()
args = Bunch(**{
    "adam_beta1" : 0.9,
    "adam_beta2" : 0.999,
    "always_fixbase" : False,
    "arch" : "resnet50",
    "cuhk03_classic_split" : False,
    "cuhk03_labeled" : False,
    "eval_freq" : -1,
    "evaluate" : True,
    "fixbase_epoch" : 0,
    "gamma" : 0.1,
    "gpu_devices" : "0",
    "height" : 128,
    "htri_only" : False,
    "label_smooth" : False,
    "lambda_htri" : 1,
    "lambda_xent" : 1,
    "load_weights" : "log/resnet50-market1501-xent/checkpoint_ep40.pth.tar",
    "lr" : 0.0003,
    "margin" : 0.3,
    "max_epoch" : 60,
    "momentum" : 0.9,
    "num_instances" : 4,
    "open_layers" : ["classifier"],
    "optim" : "adam",
    "pool_tracklet_features" : "avg",
    "print_freq" : 10,
    "resume" : "",
    "rmsprop_alpha" : 0.99,
    "root" : "data",
    "sample_method" : "evenly",
    "save_dir" : "log/eval-resnet50-market1501-xent",
    "seed" : 1,
    "seq_len" : 15,
    "sgd_dampening" : 0,
    "sgd_nesterov" : False,
    "source_names" : ["market1501"],
    "split_id" : 0,
    "start_epoch" : 0,
    "start_eval" : 0,
    "stepsize" : [20, 40],
    "target_names" : ["market1501"],
    "test_batch_size" : 100,
    "train_batch_size" : 32,
    "train_sampler" : "",
    "use_avai_gpus" : False,
    "use_cpu" : False,
    "use_metric_cuhk03" : False,
    "visualize_ranks" : True,
    "weight_decay" : 0.0005,
    "width" : 64,
    "workers" : 4
})

class MyImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = read_image(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, img_path

class Classify():
    def __init__(self):
        global args
        use_gpu = False
        print("==========\nArgs:{}\n==========".format(args))

        print("Initializing image data manager")
        self.dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
        # ------------
        self.galleryloader = self.dm.return_galleryloader(args.target_names[0])
        print("Initializing model: {}".format(args.arch))
        self.model = models.init_model(name=args.arch, num_classes=self.dm.num_train_pids, loss={'xent'}, use_gpu=use_gpu)
        print("Model size: {:.3f} M".format(count_num_param(self.model)))

        if args.load_weights and check_isfile(args.load_weights):
            # load pretrained weights but ignore layers that don't match in size
            checkpoint = torch.load(args.load_weights)
            pretrain_dict = checkpoint['state_dict']
            model_dict = self.model.state_dict()
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrain_dict)
            self.model.load_state_dict(model_dict)
            print("Loaded pretrained weights from '{}'".format(args.load_weights))

        self.gf, self.g_pids, self.g_camids = self.init_gallary_info()

       
    
    def init_gallary_info(self, use_gpu=False):
        batch_time = AverageMeter()
        # init gallary info 
        gf, g_pids, g_camids = [], [], []
        pkl_path = './data/market1501/market1501.pkl'
        if not check_isfile(pkl_path):
            end = time.time()
            for batch_idx, (imgs, pids, camids, _) in enumerate(self.galleryloader):
                end = time.time()
                features = self.model(imgs)
                batch_time.update(time.time() - end)

                features = features.data.cpu()
                gf.append(features)
                g_pids.extend(pids)
                g_camids.extend(camids)
            
            gf = torch.cat(gf, 0)
            g_pids = np.asarray(g_pids)
            g_camids = np.asarray(g_camids)
            print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
            with open(pkl_path, mode='wb') as fout:
                pickle.dump(gf, fout)
                pickle.dump(g_pids, fout)
                pickle.dump(g_camids, fout)
        else:
            with open(pkl_path, mode='rb') as fin:
                gf = pickle.load(fin)
                g_pids = pickle.load(fin)
                g_camids = pickle.load(fin)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        return gf, g_pids, g_camids
    
    def create_quereyloader(self, img_paths):
        print(img_paths)
        dataset = []
        for img_path in img_paths:
            dataset.append((img_path))
        transform_test = build_transforms(args.height, args.width, is_train=False)

        return DataLoader(
                MyImageDataset(dataset, transform=transform_test),
                batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                pin_memory=args.use_cpu, drop_last=False
        )

    def infer(self, img_paths=None):
        print("Evaluate only")
        img_paths = ['./data/market1501/query/0026_c.jpg']
        queryloader = self.create_quereyloader(img_paths)

        for name in args.target_names:
            print("Evaluating {} ...".format(name))
            distmat = self.test(queryloader, return_distmat=True)
        
            save_ranked_results(
                distmat, {'query':img_paths,'gallery':self.dm.return_testdataset_gakkery(name)},
                save_dir=osp.join(args.save_dir, 'ranked_results', name),
                topk=20
            )

    def test(self, queryloader, ranks=[1, 5, 10, 20], use_gpu=False, return_distmat=False):
        batch_time = AverageMeter()
        
        self.model.eval()

        with torch.no_grad():
            qf, q_pids, q_camids = [], [], []
            for batch_idx, (imgs, _) in enumerate(queryloader):
                if use_gpu: imgs = imgs.cuda()

                end = time.time()
                features = self.model(imgs)
                batch_time.update(time.time() - end)
                
                features = features.data.cpu()
                qf.append(features)
            qf = torch.cat(qf, 0)
            print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        print("=> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch_size))

        m, n = qf.size(0), self.gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(self.gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, self.gf.t())
        distmat = distmat.numpy()

        return distmat
        


def main():
    classify = Classify()
    classify.infer()

def test():
    def r():
        for y in range(10):
            for x in range(10):
                yield x
    
    # for b, x in enumerate(r()):
    #     print(x)

if __name__ == '__main__':
    main()
    # print(args.use_cpu)
