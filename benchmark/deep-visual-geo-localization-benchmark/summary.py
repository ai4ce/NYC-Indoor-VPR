import os
import util
import cv2
from tqdm import tqdm

folder_path = '/scratch/ds5725/VPR-datasets-downloader/datasets/nyu-vpr/images/test/queries'

file_list = os.listdir(folder_path)

full_path_list = [os.path.join(folder_path, filename) for filename in file_list]
