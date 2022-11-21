import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


image = Image.open('/data1/dataset/rvc2022/depth/datasets_mpi_sintel/train/alley_2/image_01/0000000001.png', 'w')

height = image.height
width = image.width
top_margin = int(height - 352)
                
left_margin = int((width - 1216) / 2)  #mpi: -96
                
# image = image[top_margin:top_margin + 352, :, :]
image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
image.save('123.png')