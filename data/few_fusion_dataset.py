import os.path
import random

import torch
from PIL import Image, ImageFilter

from data.base_dataset import BaseDataset, transform_few_with_label
from data.image_folder import make_dataset


class FewFusionDataset(BaseDataset):
    """Few shot dataset
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_ABC = os.path.join(opt.dataroot, opt.phase)
        self.ABC_paths = sorted(make_dataset(self.dir_ABC))
        self.shuffled_alphas = ['0', '1', '2', '3', '4', '5', '6', '7']

        with open(os.path.join(opt.dataroot, "few_dict.txt")) as f:
            self.few_dict = f.readlines()
        assert(self.opt.nencode < self.opt.few_size)
        assert(self.opt.few_size <= len(self.shuffled_alphas))
        self.few_alphas = self.shuffled_alphas[:self.opt.few_size]

    def __getitem__(self, index):
        ABC_path = self.ABC_paths[index]
        ABC = Image.open(ABC_path).convert('RGB')
        w3, h = ABC.size
        w = int(w3 / 3)
        A = ABC.crop((0, 0, w, h))
        B = ABC.crop((w, 0, w+w, h))
        C = ABC.crop((w+w, 0, w+w+w, h))
        Bases = []
        Shapes = []
        Colors = []
        Style_paths = []
        # gaussion filter
        blur_Shapes = []
        blur_Colors = []

        ABC_path_list = list(ABC_path)
        target_font = int(ABC_path.split("/")[-1].split("_")[0])
        target_char = ABC_path_list[-5]
        label = 0.0
        if target_char in self.few_dict[target_font-11000].strip()[:self.opt.few_size]:
            label = 1.0
        # for shapes
        random.shuffle(self.few_alphas)
        chars_random = self.few_alphas[:self.opt.nencode]
        for char in chars_random:
            ABC_path_list[-5] = char  # /path/to/img/XXXX_X.png
            phase_path = "".join(ABC_path_list)
            style_path = phase_path.replace(self.opt.phase, 'style')
            Style_paths.append(style_path)
            Bases.append(Image.open(style_path).convert('RGB').crop((0, 0, w, h)))
            Shapes.append(Image.open(style_path).convert('RGB').crop((w, 0, w+w, h)))
            Colors.append(Image.open(style_path).convert('RGB').crop((w+w, 0, w+w+w, h)))

            blur_Shapes.append(
                Image.open(style_path).convert('RGB').crop((w, 0, w+w, h)).filter(
                    ImageFilter.GaussianBlur(radius=(random.random()*2+2)))
                )

            blur_Colors.append(
                Image.open(style_path).convert('RGB').crop((w+w, 0, w+w+w, h)).filter(
                    ImageFilter.GaussianBlur(radius=(random.random()*2+2)))
                )

        A, B, B_G, C, C_G, C_l, label, Bases, Shapes, Colors, blur_Shapes, blur_Colors = \
            transform_few_with_label(self.opt, A, B, C, label, Bases, Shapes, Colors, blur_Shapes, blur_Colors)

        # A is the reference, B is the gray shape, C is the gradient
        return {'A': A, 'B': B, 'B_G': B_G, 'C': C, 'C_G': C_G, 'C_l': C_l, 'label': label,
                'Bases': Bases, 'Shapes': Shapes, 'Colors': Colors,
                'blur_Shapes': blur_Shapes, 'blur_Colors': blur_Colors,
                'ABC_path': ABC_path, 'Style_paths': Style_paths,
                }

    def __len__(self):
        return len(self.ABC_paths)

    def name(self):
        return 'FewFusionDataset'

    def gaussian_filter(self, array):
        pics = list()
        N, _, _, _ = array.shape

        for i in range(N):
            arr = array[i].cpu()
            pic = Image.fromarray(arr.numpy().astype('uint8'))
            pic = pic.filter(ImageFilter.GaussianFilter(radius=(random.random()*2+2)))
            pics.append(torch.tensor(pic))

        pics = torch.cat(pics).to(self.device)
        return pics
