import torch.utils.data as data

from PIL import Image
import os
import os.path
from torchvision import transforms
import torch
from .datasets import build_dataset
_VTAB_LIST = (
    'cifar',
    'caltech101',
    'dtd',
    'oxford_flowers102',
    'oxford_iiit_pet',
    'svhn',
    'sun397',
    'patch_camelyon',
    'eurosat',
    'resisc45',
    'diabetic_retinopathy',
    'clevr_count',
    'clevr_dist',
    'dmlab',
    'kitti',
    'dsprites_loc',
    'dsprites_ori',
    'smallnorb_azi',
    'smallnorb_ele',
)
_DATASET_NAME = (
    'cifar',
    'caltech101',
    'dtd',
    'oxford_flowers102',
    'oxford_iiit_pet',
    'svhn',
    'sun397',
    'patch_camelyon',
    'eurosat',
    'resisc45',
    'diabetic_retinopathy',
    'clevr_count',
    'clevr_dist',
    'dmlab',
    'kitti',
    'dsprites_loc',
    'dsprites_ori',
    'smallnorb_azi',
    'smallnorb_ele',
    'stanford_cars',
    'oxford_flowers',
    'food-101',
    'oxford_pets',
    'fgvc_aircraft',
    'imagenet'
)
_CLASSES_NUM = (100, 102, 47, 102, 37, 10, 397, 2, 10, 45, 5, 8, 6, 6, 4, 16, 16, 18, 9, 196, 102, 101, 37, 100, 1000)

def get_classes_num(dataset_name):
    dict_ = {name: num for name, num in zip(_DATASET_NAME, _CLASSES_NUM)}
    return dict_[dataset_name]

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist

class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


def get_data(name, evaluate=False, batch_size=64, args=None):
    if name in _VTAB_LIST:
        root = 'data/vtab-1k/' + name
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if evaluate:
            train_loader = torch.utils.data.DataLoader(
                ImageFilelist(root=root, flist=root + "/train800val200.txt",
                            transform=transform),
                batch_size=batch_size, shuffle=True, drop_last=True,
                num_workers=8, pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                ImageFilelist(root=root, flist=root + "/test.txt",
                            transform=transform),
                batch_size=256, shuffle=False,
                num_workers=8, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                ImageFilelist(root=root, flist=root + "/train800.txt",
                            transform=transform),
                batch_size=batch_size, shuffle=True, drop_last=True,
                num_workers=8, pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                ImageFilelist(root=root, flist=root + "/val200.txt",
                            transform=transform),
                batch_size=256, shuffle=False,
                num_workers=8, pin_memory=True)
        return train_loader, val_loader
    else:
        dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
        dataset_val, _ = build_dataset(is_train=False, args=args)
        train_loader = torch.utils.data.DataLoader(dataset_train,
                batch_size=batch_size, shuffle=True, drop_last=True,
                num_workers=8, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(dataset_val,
                batch_size=256, shuffle=False,
                num_workers=8, pin_memory=True)
        return train_loader, val_loader

