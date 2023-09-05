import os
import ntpath
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import torch

class fusedAfrogrefineJdataset(BaseDataset):
    """
    This dataset class can load paired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--gt_prefix', type=str, default='', help='name of the used prior')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        #self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_A_frog = os.path.join(opt.dataroot, opt.phase + 'A_N2N')  # create a path '/path/to/data/trainA_frog'
        self.dir_A_refineJ = os.path.join(opt.dataroot, opt.phase + 'A_refineJ')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        # self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        # self.B_size = len(self.B_paths)  # get the size of dataset B

        # btoA = self.opt.direction == 'BtoA'
        # input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        # output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        #self.transform = get_transform(self.opt, grayscale=(self.opt.input_nc == 1))
        self.toTensor = transforms.ToTensor()
        # self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index]  # make sure index is within then range

        A_name = os.path.splitext(ntpath.basename(A_path))[0]
        #B_shortPath = '%s.jpg' % (A_name.split('_')[0])  # '%s.png'%A_name.split('_')[0]

        #B_shortPath = '%s%s.jpeg'%('_'.join(A_name.split('_')[:-1]), self.opt.gt_prefix) # '%s.png'%A_name.split('_')[0]
        #B_path = os.path.join(self.dir_B, B_shortPath)
        A_frog_path = os.path.join(self.dir_A_frog, '%s.jpg' % (A_name))
        A_refineJ_path = os.path.join(self.dir_A_refineJ, '%s.jpg' % (A_name))
        """print("A_path, A_name")
        print(A_path, A_name)
        print("B_shortPath, B_path")
        print(B_shortPath,B_path)
        print(A_frog_path)
        print('A_frog_path')"""
        A_img = Image.open(A_path).convert('RGB')
        A_frog_img = Image.open(A_frog_path).convert('RGB')
        A_refineJ_img = Image.open(A_refineJ_path).convert('RGB')

        # apply image transformation
        #A = self.transform(A_img)
        # B = self.toTensor(B_img)
        A = self.toTensor(A_img)
        A_frog = self.toTensor(A_frog_img)
        A_refineJ = self.toTensor(A_refineJ_img)
        # return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        return {'A': A, 'A_frog':A_frog,'A_refineJ':A_refineJ, 'A_paths': A_path, 'A_frog_paths': A_frog_path,
                'A_refineJ_path': A_refineJ_path,'haze': A,'paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.A_size

    def cropImage(self, img, target_size):
        ow, oh = img.size
        tw, th = target_size

        if (ow > tw or oh > th):
            x1 = np.floor((ow - tw)/2)
            y1 = np.floor((oh - th)/2)
            return img.crop((x1, y1, x1 + tw, y1 + th))
        return img

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True