'''
  celebA dataset
'''
import os
import os.path
import torch
import torch.utils.data as data
from torchvision import models
from torchvision import datasets
from torchvision import transforms
from torchvision import utils
# import torchvision.datasets as dset
# import torchvision.utils as vutils

from PIL import Image


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

#def accimage_loader(path):
#    import accimage
#    try:
#        return accimage.Image(path)
#    except IOError:
#        #Potentially a decoding problem, fall back to PIL.Image
#        return pil_loader(path)
#
#
#def default_loader(path):
#    from torchvision import get_image_backend
#    if get_image_backend() == 'accimage':
#        return accimage_loader(path)
#    else:
#        return pil_loader(path)

class celebA(data.Dataset):

    def __init__(self, root, split='train', transform=None, target_transform=None, loader=pil_loader, download=False):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images i subfolders of: " + root + "\n"
                               "Supported image ext. are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
