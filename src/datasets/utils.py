import torch
import random
import torchvision.transforms as transforms


def resize(img, scale):
    """
    :param img: (C, H, W)
    :param scale: int
    :return: img_down: (C, H//scale, W//scale)
    """
    assert scale in [2, 4], "当前只支持 scale为2或4 的下采样"
    C, H, W = img.shape
    if H % 4 != 0 or W % 4 != 0:
        H -= H % 4
        W -= W % 4
        img = img[:, :H, :W]
    resize_fn = transforms.Resize((H // scale, W // scale), transforms.InterpolationMode.BICUBIC)
    img_down = resize_fn(img)
    return img, img_down


def patchify(img, patch_size, return_coords=False):
    """
    :param img: (C, H, W)
    :param patch_size: int
    :return: (C, patch_size, patch_size)
    """
    C, H, W = img.shape
    x0 = random.randint(0, H - patch_size)
    y0 = random.randint(0, W - patch_size)
    img_cropped = img[:, x0: x0 + patch_size, y0: y0 + patch_size]
    if return_coords:
        return img_cropped, x0, y0
    else:
        return img_cropped


def encode_mwd(*imgs):
    """
    :param img: each shape:(1, H, W)
    :return: each shape:(2, H, W)
    """
    res = []
    for img in imgs:
        img = torch.clip(img, min=0, max=360)
        img = torch.deg2rad(img)
        cos_img, sin_img = torch.cos(img), torch.sin(img)
        img = torch.cat([cos_img, sin_img], dim=0)
        res.append(img)
    return res[0] if len(res) == 1 else tuple(res)


def augement(*imgs):
    """
    :param imgs: each shape: (C, H, W)
    :return: each shape: (C, H, W)
    """
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    dflip = random.random() < 0.5

    res = []
    for img in imgs:
        if hflip:
            img = img.flip(-2)
        if vflip:
            img = img.flip(-1)
        if dflip:
            img = img.transpose(-2, -1)
        res.append(img)
    return res[0] if len(res) == 1 else tuple(res)


def normalize(mean, std, *imgs):
    """
    :param img: (C, H, W)
    :param mean: (C, 1, 1)
    :param std: (C, 1, 1)
    :return: (C, H, W)
    """
    res = []
    for img in imgs:
        img = (img - mean) / std
        res.append(img)
    return res[0] if len(res) == 1 else tuple(res)
