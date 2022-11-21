from glob import glob
import os
import glob
from PIL import Image
import sintel_io

depth_path = '/data1/dataset/rvc2022/depth/datasets_mpi_sintel/train/alley_2/proj_depth/groundtruth/image_01_dpt/frame_0001.dpt'  


depth_gt = Image.fromarray(sintel_io.depth_read(depth_path))

depth_gt.save("1111111.jpg")