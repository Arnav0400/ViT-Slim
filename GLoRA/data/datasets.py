
import os
import json
import torch
import scipy
import scipy.io as sio
import glob
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

class general_dataset_few_shot(ImageFolder):
    def __init__(self, root, dataset,train=True, transform=None, target_transform=None, mode=None,is_individual_prompt=False,shot=2,seed=0,**kwargs):
        self.dataset_root = root
        self.dataset = dataset
        if self.dataset == 'imagenet':
            self.dataset_root = 'data/imagenet'
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform
        
        if mode == 'super':
            train_list_path = os.path.join(self.dataset_root, 'annotations/train_meta.list.num_shot_'+str(shot)+'.seed_'+str(seed))
        elif mode == 'search':
            if 'imagenet' in self.dataset_root:
                train_list_path = os.path.join(self.dataset_root, 'annotations/unofficial_val_list_4_shot16seed0')
            else:
                train_list_path = os.path.join(self.dataset_root, 'annotations/val_meta.list')
        else:
            if 'imagenet' in self.dataset_root and self.dataset != 'imagenet':
                train_list_path = os.path.join(self.dataset_root, 'annotations/val_meta.list')
            else:
                train_list_path = os.path.join(self.dataset_root, 'annotations/train_meta.list.num_shot_'+str(shot)+'.seed_'+str(seed))

        if mode == 'search' and not 'imagenet' in self.dataset_root:
            test_list_path = os.path.join(self.dataset_root, 'annotations/train_meta.list.num_shot_1.seed_0')           
        elif 'imagenet' in self.dataset_root:
            test_list_path = os.path.join(self.dataset_root, 'annotations/val_meta.list')
        else:
            test_list_path = os.path.join(self.dataset_root, 'annotations/test_meta.list')

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.rsplit(' ',1)[0]
                    label = int(line.rsplit(' ',1)[1])
                    if 'stanford_cars' in self.dataset_root or ('imagenet' in self.dataset_root and 'imagenet' != self.dataset):
                        self.samples.append((os.path.join(self.dataset_root,img_name), label))
                    elif 'imagenet' == self.dataset:
                        self.samples.append((os.path.join(self.dataset_root+'/train',img_name), label))
                    else:
                        self.samples.append((os.path.join(self.dataset_root+'/images',img_name), label))
        elif self.dataset=='imagenet':
            label_dict = {}
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.rsplit(' ',1)[0]
                    label = int(line.rsplit(' ',1)[1])
                    label_dict[img_name] = label
            for img in glob.glob(self.dataset_root+'/val/*/*.JPEG'):
                self.samples.append((img,label_dict[img.split('/')[-1]]))
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.rsplit(' ',1)[0]
                    label = int(line.rsplit(' ',1)[1])
                    if 'stanford_cars' in self.dataset_root or ('imagenet' in self.dataset_root and 'imagenet' != self.dataset):
                        self.samples.append((os.path.join(self.dataset_root,img_name), label))
                    elif 'imagenet' == self.dataset:
                        if mode == 'search':
                            self.samples.append((os.path.join(self.dataset_root+'/train',img_name), label))
                        else:
                            self.samples.append((os.path.join(self.dataset_root+'/val',img_name), label))
                    else:
                        self.samples.append((os.path.join(self.dataset_root+'/images',img_name), label))


def build_dataset(is_train, args, folder_name=None):
    transform = build_transform(is_train, args.mode)
    dataset = general_dataset_few_shot(f'data/{args.dataset}', args.dataset,train=is_train, transform=transform,mode=args.mode,shot=args.few_shot_shot,seed=args.few_shot_seed)
    if 'stanford_cars' in args.dataset:
        nb_classes = 196
    elif 'oxford_flowers' in args.dataset:
        nb_classes = 102
    elif 'food-101' in args.dataset:
        nb_classes = 101
    elif 'oxford_pets'in args.dataset:
        nb_classes = 37
    elif 'fgvc_aircraft' in args.dataset:
        nb_classes = 100
    elif 'imagenet' in args.dataset:
        nb_classes = 1000

    return dataset, nb_classes

def build_transform(is_train, mode):
    if is_train and mode != 'search':
        transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
        return transform

    t = []
    size = int((256 / 224) * 224)

    t.append(
        transforms.Resize((size,size), interpolation=3)  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(224))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)