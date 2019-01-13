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
args = Bunch(**{
    "arch" : "resnet50",
    "num_train_pids": 751,
    "topk": 10,
    "height" : 128,
    "width" : 64,
    "load_weights" : "log/market1501-xent-htri/checkpoint_ep60.pth.tar",
    "save_dir" : "log/",
    "gallery_path": './data/market1501/bounding_box_test',
    "seed" : 1,
    "test_batch_size" : 128,
    "train_batch_size" : 128,
    "root" : "data",
    "split_id" : 0,
    "workers" : 4,
    "train_sampler" : "",
    "num_instances" : 4,
    "cuhk03_classic_split" : False,
    "cuhk03_labeled" : False,
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


torch.manual_seed(args.seed)
use_gpu = torch.cuda.is_available()
if use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print("Currently using GPU {}".format(1))
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)

class Classify():
    def __init__(self, gallery_path=args.gallery_path, save_dir=args.save_dir):
        print("==========\nArgs:{}\n==========".format(args))
        self.gallery_path = gallery_path
        self.save_dir=save_dir

        print("gallery_path".format(gallery_path))
        print("Initializing image data manager")
        # ------------
        self.galleryloader, self.gallery_datasets = self.create_galleryloader()
        print("Initializing model: {}".format(args.arch))
        self.model = models.init_model(name=args.arch, num_classes=args.num_train_pids, loss={'xent', 'htri'})
        print("Model size: {:.3f} M".format(count_num_param(self.model)))

        if args.load_weights and check_isfile(args.load_weights):
            # load pretrained weights but ignore layers that don't match in size
            map_location = 'cpu' if not use_gpu else None
            checkpoint = torch.load(args.load_weights, map_location=map_location)
            pretrain_dict = checkpoint['state_dict']
            model_dict = self.model.state_dict()
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrain_dict)
            self.model.load_state_dict(model_dict)
            print("Loaded pretrained weights from '{}'".format(args.load_weights))
        if use_gpu:
            self.model = nn.DataParallel(self.model).cuda()
        self.gf = self.init_gallary_info()

    
    def init_gallary_info(self):
        batch_time = AverageMeter()
        # init gallary info 
        self.model.eval()
        with torch.no_grad():
            gf, g_pids, g_camids = [], [], []
            save_type = 'gpu' if use_gpu else 'cpu'
            pkl_path = './data/market1501/market1501-%s.pkl' % save_type
            if not check_isfile(pkl_path):
                end = time.time()
                for batch_idx, (imgs, _) in enumerate(self.galleryloader):
                    if use_gpu: 
                        imgs = imgs.cuda()
                    end = time.time()
                    features = self.model(imgs)
                    batch_time.update(time.time() - end)

                    features = features.data.cpu()
                    gf.append(features)
            
                gf = torch.cat(gf, 0)
                # cache for CPU
                with open(pkl_path, mode='wb') as fout:
                    pickle.dump(gf, fout)
            else:
                with open(pkl_path, mode='rb') as fin:
                    gf = pickle.load(fin)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        return gf
    
    def create_quereyloader(self, img_paths):
        dataset = []
        for img_path in img_paths:
            dataset.append((img_path))
        transform_test = build_transforms(args.height, args.width, is_train=False)

        return DataLoader(
                MyImageDataset(dataset, transform=transform_test),
                batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                pin_memory=False, drop_last=False
        )

    def create_galleryloader(self):
        '''Create self-gallery datasets use gallery_path'''
        dataset = []
        img_paths = glob.glob(osp.join(self.gallery_path, '*.jpg'))
        for img_path in img_paths:
            dataset.append((img_path))
        transform_test = build_transforms(args.height, args.width, is_train=False)

        galleryloader = DataLoader(
                MyImageDataset(dataset, transform=transform_test),
                batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                pin_memory=False, drop_last=False
        )
        return galleryloader, dataset

    def infer(self, img_paths=None):
        '''Infer 

        Args:
            img_paths(list): Image path.
        Returns:
            List of result path.
        '''

        queryloader = self.create_quereyloader(img_paths)
        save_dirs = []
        # for name in args.target_names:
        distmat = self.test(queryloader, return_distmat=True)
        save_dir = osp.join(self.save_dir, 'ranked_results')
        save_ranked_results(
            distmat, {'query':img_paths,'gallery':self.gallery_datasets},
            save_dir=save_dir,
            topk=args.topk
        )
        save_dirs.append(save_dir)
        return save_dirs

    def test(self, queryloader, return_distmat=False):
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

        m, n = qf.size(0), self.gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(self.gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, self.gf.t())
        distmat = distmat.numpy()

        return distmat
        
def main():
    #classify = Classify(gallery_path='./data/video_data/video/')
    classify = Classify(gallery_path='./data/market1501/bounding_box_test')
    import glob
    img_paths = glob.glob(osp.join('./data/market1501/images', '*.jpg'))
    #img_paths = ['./data/market1501/images/0026_c.jpg']
    print(classify.infer(img_paths))


if __name__ == '__main__':
    main()
