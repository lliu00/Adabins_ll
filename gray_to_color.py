import glob
import os
import matplotlib
from tqdm import tqdm
import cv2
import torch
import numpy as np

# color the depth, kitti magma_r, nyu jet
def colorize(value, cmap='magma_r', vmin=None, vmax=None):
    # TODO: remove hacks

    # for abs
    # vmin=1e-3
    # vmax=80

    # for relative
    # value[value<=vmin]=vmin

    # vmin=None
    # vmax=None

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True) # ((1)xhxwx4)
    # value = torch.tensor(value).unsqueeze(0)
    value = value[:, :, :, :3] # bgr -> rgb
    rgb_value = value[..., ::-1]

    return rgb_value

file_sintel = open("/ssd/szy/test/rabbitAI_vis_list.txt", "r")
img_names = file_sintel.readlines()
# img_names = glob.glob(os.path.join('viper_gt', "*"))
num_images = len(img_names)

os.makedirs('scaled_result_max100_color', exist_ok=True)


for path in tqdm(img_names):
    path = path.replace("\n", "")
    pre_depth = cv2.imread(path, -1) / 256
    h, w = pre_depth.shape
    # if(path.split("/")[1] == "023_im0.png"):
    #     np.savetxt("flower.csv", pre_depth.astype(np.uint8), delimiter=",")
    pre_depth_color =  colorize(pre_depth.reshape(1,h,w))
    
    cv2.imwrite('scaled_result_max100_color' + "/" + path.split("/")[11], pre_depth_color.reshape(h, w, 3))

    # pre_depth_color = pd.DataFrame(pre_depth_color)
    # pre_depth_color
    