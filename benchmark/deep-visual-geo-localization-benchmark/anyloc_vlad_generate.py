# Download cache data from OneDrive
import os
from onedrivedownloader import download
from utilities1 import od_down_links

# Link
ln = od_down_links["cache"]
# Download and unzip
if os.path.isdir("./cache"):
    print("Cache folder already exists!")
else:
    print("Downloading the cache folder")
    download(ln, filename="cache.zip", unzip=True, unzip_path="./")
    print("Cache folder downloaded")

import glob
_ex = lambda x: os.path.realpath(os.path.expanduser(x))
cache_dir: str = _ex("./cache")
# imgs_dir = "/mnt/data/dean/datasets/b4/images/test/database"
# assert os.path.isdir(cache_dir), "Cache directory not found"
# assert os.path.isdir(imgs_dir), "Invalid unzipping"
# num_imgs = len(glob.glob(f"{imgs_dir}/*.jpg"))
# print(f"Found {num_imgs} images in {imgs_dir}")

# Import everything
import numpy as np
import cv2 as cv
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as tvf
from torchvision.transforms import functional as T
from PIL import Image
import matplotlib.pyplot as plt
import distinctipy as dipy
from tqdm.auto import tqdm
from typing import Literal, List
import os
import natsort
import shutil
from copy import deepcopy
# DINOv2 imports
from utilities1 import DinoV2ExtractFeatures
from utilities1 import VLAD

# Program parameters
save_dir = "/home/unav/Desktop/benchmark/AnyLoc/saved_desc"
device = torch.device("cuda")
# Dino_v2 properties (parameters)
desc_layer: int = 31
desc_facet: Literal["query", "key", "value", "token"] = "value"
num_c: int = 32
# Domain for use case (deployment environment)
domain: Literal["aerial", "indoor", "urban"] = "urban"
# Maximum image dimension
max_img_size: int = 640

# DINO extractor
if "extractor" in globals():
    print(f"Extractor already defined, skipping")
else:
    # extractor=ViTExtractor("dino_vits8", stride=4, 
    #     device=device)
    extractor = DinoV2ExtractFeatures("dinov2_vitg14", desc_layer,
        desc_facet, device=device)
# Base image transformations
base_tf = tvf.Compose([
    tvf.ToTensor(),
    tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
])

# Ensure that data is present
ext_specifier = f"dinov2_vitg14/l{desc_layer}_{desc_facet}_c{num_c}"
c_centers_file = os.path.join(cache_dir, "vocabulary", ext_specifier,
                            domain, "c_centers.pt")
assert os.path.isfile(c_centers_file), "Cluster centers not cached!"
c_centers = torch.load(c_centers_file)
assert c_centers.shape[0] == num_c, "Wrong number of clusters!"

# VLAD object
vlad = VLAD(num_c, desc_dim=None, 
        cache_dir=os.path.dirname(c_centers_file))
# Fit (load) the cluster centers (this'll also load the desc_dim)
vlad.fit(None)


# img_fnames = glob.glob(f"{imgs_dir}/*.jpg")
# img_fnames = natsort.natsorted(img_fnames)

def single_desc(img_fname):
#     for img_fname in tqdm(img_fnames[:20]):
#         # DINO features
    with torch.no_grad():
        pil_img = Image.open(img_fname).convert('RGB')
        img_pt = base_tf(pil_img).to(device)
        if max(img_pt.shape[-2:]) > max_img_size:
            c, h, w = img_pt.shape
            # Maintain aspect ratio
            if h == max(img_pt.shape[-2:]):
                w = int(w * max_img_size / h)
                h = max_img_size
            else:
                h = int(h * max_img_size / w)
                w = max_img_size
            # print(f"To {(h, w) =}")
            img_pt = T.resize(img_pt, (h, w), 
                    interpolation=T.InterpolationMode.BICUBIC)
            # print(f"Resized {img_fname} to {img_pt.shape = }")
        # Make image patchable (14, 14 patches)
        c, h, w = img_pt.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None, ...]
        # Extract descriptor
        # print(img_pt.shape)
        ret = extractor(img_pt) # [1, num_patches, desc_dim]
    # VLAD global descriptor
    gd = vlad.generate(ret.cpu().squeeze()) # VLAD: shape [agg_dim]
    # print(gd.shape)
    return gd
    # gd_np = gd.numpy()[np.newaxis, ...] # shape: [1, agg_dim]
    # print(gd_np.shape)
        # np.save(f"{save_dir}/{os.path.basename(img_fname)}.npy", gd_np)

# single_desc("/mnt/data/nyc_indoor/indoor/images/test/database/@00000.00@00035.70@168@.jpg")
def all_desc():
    f=open("test_database.txt")
    f1=open("test_queries.txt")
    l=f.readlines()
    l1=f1.readlines()
    all_features = np.empty((len(l)+len(l1), 49152), dtype="float32")
    for i in tqdm(range(len(l))):
        all_features[i]=single_desc(l[i].strip())

    for i in tqdm(range(len(l1))):
        all_features[len(l)+i]=single_desc(l1[i].strip())
    return all_features