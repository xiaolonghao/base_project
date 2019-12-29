# -*- coding: utf-8 -*-
import random
import math
from PIL import Image
from torchvision.transforms.transforms import *

OftenUsedTransforms = '''Compose':"Composes several transforms together",
                       'Resize':"Resize the input PIL Image to the given size",
                       'RandomCrop':"Crop the given PIL Image at a random location",
                       'RandomHorizontalFlip':"Horizontally flip the given PIL Image randomly with a given probability",
                       'RandomResizedCrop':"Randomly cropped and resized image."
                        'ToTensor':"Convert a  PIL Image  or  numpy  to tensor"
                       'Normalize':"Normalize a tensor image with mean and standard deviation",
                        'IncreaseRandomCrop': 'first increase image size to (1 + 1/8), and then perform random crop.'
                        'RandomErasing':"Random erasing"
                        '''

class IncreaseRandomCrop(object):
    """
    先resize 在 crop
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
    - height (int): target height.
    - width (int): target width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(round(self.width * 1.1)), int(round(self.height * 1.1))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img

class RandomErasing(object):
    """
    随机擦除算法实现
    erase_max_ratio: The probability that the Random Erasing operation will be performed.
    sl: Minimum proportion of erased area against input image.
    sh: Maximum proportion of erased area against input image.
    r1: Minimum aspect ratio of erased area.
    means: Erasing value.
    注意：此增强算法放在toTensor之后
    """

    def __init__(self,erase_max_ratio=0.5,sl = 0.02, sh = 0.4, r1 = 0.3,means=(0.5,0.5,0.5)):
        self.erase_max_ratio = erase_max_ratio
        self.means = means
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.erase_max_ratio:
            return img
        for attempt in range(100):
            # img_size为CxHxW
            img_size = img.size()
            areas = img_size[1]*img_size[2]
            # 以下超参数为论文中所选
            target_area = random.uniform(self.sl, self.sh) * areas
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(pow(target_area*aspect_ratio,0.5)))
            w = int(round(pow(target_area/aspect_ratio,0.5)))

            if h < img_size[1] and w < img_size[2]:
                x = random.randint(0,img_size[1]-h)
                y = random.randint(0,img_size[2]-w)
                img[0,x:x+h,y:y+w] = self.means[0]
                img[1,x:x+h,y:y+w] = self.means[1]
                img[2,x:x+h,y:y+w] = self.means[2]
                return img
        return img

# 以下用的少
class RectResize(object):
    """
    就是resize
    """
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)

class RectCropAndResize(object):
    """
    crop and resize
    """
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)
        # Fallback
        scale = RectResize(self.height, self.width,interpolation=self.interpolation)
        return scale(img)
