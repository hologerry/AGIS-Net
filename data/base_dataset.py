import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'none':
        transform_list = []

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomVerticalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def transform_multi(opt, A, B, C, Bases, Shapes, Colors, blur_Shapes, blur_Colors):
    """Transformer for multi fusion, pretrain dataset
    """
    if not opt.resize_or_crop == 'none':
        raise ValueError(
            "Only support none mode for resize_or_crop on base_gray_color dataset")
    assert(isinstance(Bases, list))
    assert(isinstance(Shapes, list))
    assert(isinstance(Colors, list))
    assert(isinstance(blur_Shapes, list))
    assert(isinstance(blur_Colors, list))
    A = transforms.ToTensor()(A)
    B = transforms.ToTensor()(B)
    C = transforms.ToTensor()(C)
    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
    B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
    C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C)

    Bases = list(map(lambda b: transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(b)), Bases))
    Shapes = list(map(lambda s: transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(s)), Shapes))
    Colors = list(map(lambda c: transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(c)), Colors))
    blur_Shapes = list(map(lambda bs: transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(bs)), blur_Shapes))
    blur_Colors = list(map(lambda bc: transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(bc)), blur_Colors))

    Bases = torch.cat(Bases)
    Shapes = torch.cat(Shapes)
    Colors = torch.cat(Colors)
    blur_Shapes = torch.cat(blur_Shapes)
    blur_Colors = torch.cat(blur_Colors)

    return A, B, C, Bases, Shapes, Colors, blur_Shapes, blur_Colors


def transform_few_with_label(opt, A, B, C, label, Bases, Shapes, Colors, blur_Shapes, blur_Colors):
    if not opt.resize_or_crop == 'none':
        raise ValueError(
            "Only support none mode for resize_or_crop on base_gray_color dataset")
    assert(isinstance(Bases, list))
    assert(isinstance(Shapes, list))
    assert(isinstance(Colors, list))
    assert(isinstance(blur_Shapes, list))
    assert(isinstance(blur_Colors, list))
    A = transforms.ToTensor()(A)
    B = transforms.ToTensor()(B)
    C = transforms.ToTensor()(C)
    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
    B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
    C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C)

    if label == 0.0:
        C_l = torch.zeros_like(C)
    else:
        C_l = C
    label = torch.tensor(label)

    Bases = list(map(lambda b: transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(b)), Bases))
    Shapes = list(map(lambda s: transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(s)), Shapes))
    Colors = list(map(lambda c: transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(c)), Colors))
    blur_Shapes = list(map(lambda bs: transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(bs)), blur_Shapes))
    blur_Colors = list(map(lambda bc: transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(bc)), blur_Colors))

    rand_idx = random.randrange(opt.nencode)
    if label == 0.0:
        B_G = Shapes[rand_idx]
        C_G = Colors[rand_idx]
    else:
        B_G = B
        C_G = C

    Bases = torch.cat(Bases)
    Shapes = torch.cat(Shapes)
    Colors = torch.cat(Colors)
    blur_Shapes = torch.cat(blur_Shapes)
    blur_Colors = torch.cat(blur_Colors)

    return A, B, B_G, C, C_G, C_l, label, Bases, Shapes, Colors, blur_Shapes, blur_Colors


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)


def transform_grid(Shapes, Colors):
    """Transform Shpaes/Colors to grid
       0 1 2 3 ->  0 1
                   2 3

    Arguments:
        Shapes {PIL Images} -- [4*W*H*C]
        Colors {PIL Images} -- [4*W*H*C]

    Returns:
        [grid tensors] -- [C*2W*2H]
    """

    assert(isinstance(Shapes, list))
    assert(isinstance(Colors, list))

    Shapes = list(map(lambda s: transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(s)), Shapes))
    shapes_row1 = torch.cat((Shapes[0], Shapes[1]), 1)
    shapes_row2 = torch.cat((Shapes[2], Shapes[3]), 1)
    Shapes_grid = torch.cat((shapes_row1, shapes_row2), 2)

    Colors = list(map(lambda c: transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(c)), Colors))
    colors_row1 = torch.cat((Colors[0], Colors[1]), 1)
    colors_row2 = torch.cat((Colors[2], Colors[3]), 1)
    Colors_grid = torch.cat((colors_row1, colors_row2), 2)

    return Shapes_grid, Colors_grid
