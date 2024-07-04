import os
import util
import cv2
from tqdm import tqdm

def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

folder_path = '/scratch/ds5725/VPR-datasets-downloader/datasets/indoor_new/images/val'

file_list=get_file_paths(folder_path)

full_path_list = [os.path.join(folder_path, filename) for filename in file_list]
# file_list = os.listdir(folder_path)

# full_path_list = [os.path.join(folder_path, filename) for filename in file_list]

# folder_path = '/scratch/ds5725/deep-visual-geo-localization-benchmark/visual/resnet'

# file_list = os.listdir(folder_path)

# full_path_list1 = [os.path.join(folder_path, filename) for filename in file_list]

# d={}
# f=open("/scratch/ds5725/ssl_vpr/sub/sub_test_utm.txt")
# for line in f:
#     s=line.strip().split()
#     d[s[0]]=(util.format_coord(float(s[1])),util.format_coord(float(s[2])))


# for i in tqdm(range(len(full_path_list))):
#     substr=os.path.basename(full_path_list[i])
#     substr=substr[:-6]+".jpg"
#     utm_east=d[substr][0]
#     utm_north=d[substr][1]
#     new_name=""
#     print(utm_east,utm_north)
#     for ip1 in full_path_list1:
#         print(ip1)
#         break
#         if utm_east in ip1 and utm_north in ip1:
#             new_name=os.path.basename(ip1)
#             break
#     if new_name!="":
#         img=cv2.imread(full_path_list[i])
#         cv2.imwrite("/scratch/ds5725/deep-visual-geo-localization-benchmark/visual/simclr_utm/"+new_name, img)
    
f1=open("val_paths.txt", "w")
for fp in full_path_list:
    f1.write(fp+'\n')

f2=open("val_utm.txt","w")
for line in full_path_list:
    substr=os.path.basename(line)
    first_number = float(substr.split('@')[1])
    second_number = float(substr.split('@')[2])
    f2.write(substr+" "+str(first_number)+" "+str(second_number)+'\n')

