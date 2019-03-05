#coding : utf8
#将数据随机分成训练集，验证集和测试集
import os
import glob
import random
import numpy as np
import shutil

image_path = r'E:\tree\raw_data\x256'
mask_path = r'E:\tree\raw_data\y256'

image_name_arr = np.array(glob.glob(os.path.join(image_path,"*.tif")))
mask_name_arr = np.array(glob.glob(os.path.join(mask_path,"*.tif")))
total_nb = len(image_name_arr)
idx = [i for i in range(total_nb)]
random.shuffle(idx) 
image_name_arr = image_name_arr[idx]
mask_name_arr = mask_name_arr[idx]

def mkpath(path):
    if os.path.exists(path) == False:
        os.makedirs(path)
    return path
target_path = r'E:\tree\randomdata2'
val_image_path =  mkpath(os.path.join(target_path,'val\image'))
val_mask_path =  mkpath(os.path.join(target_path,'val\label'))
test_image_path =  mkpath(os.path.join(target_path,'test\image'))
test_mask_path =  mkpath(os.path.join(target_path,'test\label'))
train_image_path =  mkpath(os.path.join(target_path,'train\image'))
train_mask_path =  mkpath(os.path.join(target_path,'train\label'))

for i in range(0,300):
    new_image_path = image_name_arr[i].replace(image_path,test_image_path)
    shutil.copy(image_name_arr[i],new_image_path)
    new_mask_path = mask_name_arr[i].replace(mask_path,test_mask_path)
    shutil.copy(mask_name_arr[i],new_mask_path)
    
for i in range(300,500):
    new_image_path = image_name_arr[i].replace(image_path,val_image_path)
    shutil.copy(image_name_arr[i],new_image_path)
    new_mask_path = mask_name_arr[i].replace(mask_path,val_mask_path)
    shutil.copy(mask_name_arr[i],new_mask_path)
    
for i in range(500,total_nb):
    new_image_path = image_name_arr[i].replace(image_path,train_image_path)
    shutil.copy(image_name_arr[i],new_image_path)
    new_mask_path = mask_name_arr[i].replace(mask_path,train_mask_path)
    shutil.copy(mask_name_arr[i],new_mask_path)
print('finish!')
