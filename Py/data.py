# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np 
import os
import glob
import skimage.io as io
#import skimage.transform as trans
#from libtiff import TIFF
import random
import keras
from keras import backend as K
from project_img import *

def adjustData(img,mask=[]):
    #将图像都归一化至0-1
    img = img/255
    new_img = img.transpose(1,2,0)
    if len(mask):
        new_mask = np.zeros(mask.shape+(1,),dtype=np.float32)
        new_mask[:,:,0] = mask / 255
        return (new_img,new_mask)
    else:
        return (new_img,None)
        
def getAllData(image_path,mask_path):
    #将所有数据都存进内存中，小数据集可以使用
    #image_name_arr = glob.glob(os.path.join(image_path,"*.tif"))
    mask_name_arr = glob.glob(os.path.join(mask_path,"*.tif"))
    image_name_arr = []
    image_arr = []
    mask_arr = []
    for i in range(len(mask_name_arr)):
        tmp_image_name_arr = mask_name_arr[i].replace(mask_path,image_path)
        tmp_image_name_arr = tmp_image_name_arr.replace('.tif','_pp.tif')
        img,mask = getData(tmp_image_name_arr,mask_name_arr[i])
        image_arr.append(adjustmask(img))
        mask_arr.append(adjustmask(mask))
        image_name_arr.append(tmp_image_name_arr)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_name_arr,mask_name_arr,image_arr,mask_arr
    
def adjustmask(mask):
    new_mask = np.zeros(mask.shape+(1,),dtype=np.float32)
    new_mask[:,:,0] = mask / 255
    return new_mask
    
def getData(image_name_arr, mask_name_arr=None):
    #由于是八波段，需要用TIFF模块读入，返回numpy数组，后来没用tiff包了，用gdal可以更好的读取遥感数据
    #tif_img = TIFF.open(image_name_arr, mode='r')
    #img = tif_img.read_image()
    _,_,img = read_img(image_name_arr)
    if mask_name_arr==None:
        return (img,[])
    else:
        #tif_mask = TIFF.open(mask_name_arr, mode='r')
        #mask = tif_mask.read_image()
        _,_,mask = read_img(mask_name_arr)
        return (img,mask)
        
def dataGenerator(image_path, mask_path, batch_size=2, image_size=(256,256,9), mask_size=(256,256,1), mode='train'):
    #数据生成器，图像太大，一次性全读入内存会爆炸，需要通过图像文件路径，逐次生成x与对应y
    image_name_arr = np.array(glob.glob(os.path.join(image_path,"*.tif")))
    mask_name_arr = np.array(glob.glob(os.path.join(mask_path,"*.tif")))
    total_nb = len(image_name_arr)
    new_img = np.zeros((batch_size,)+image_size) 
    new_mask = np.zeros((batch_size,)+mask_size) 
    idx = [i for i in range(total_nb)]
    batch_count = 0
    if mode == 'train':
        while True:
            random.shuffle(idx) 
            image_name_arr = image_name_arr[idx]
            mask_name_arr = mask_name_arr[idx]
            for i in range(total_nb):
                #print('train',i)
                img,mask = adjustData(*getData(image_name_arr[i],mask_name_arr[i]))
                new_img[batch_count,:,:,:] = img
                new_mask[batch_count,:,:,:] = mask
                batch_count += 1
                if batch_count == batch_size:
                    yield (new_img,new_mask)
                    new_img = np.zeros((batch_size,)+image_size)
                    new_mask = np.zeros((batch_size,)+mask_size)
                    batch_count = 0
    elif mode == 'test':
        for i in range(total_nb):
            #print('test',i)
            img,mask = adjustData(*getData(image_name_arr[i],mask_name_arr[i]))
            new_img[batch_count,:,:,:] = img
            new_mask[batch_count,:,:,:] = mask
            batch_count += 1
            if batch_count == batch_size:
                yield (new_img,new_mask)
                new_img = np.zeros((batch_size,)+image_size)
                new_mask = np.zeros((batch_size,)+mask_size)
                batch_count = 0
            


def predictGenerator(test_path,batch_size = 2,image_size = (256,256,9)):
    #预测时的生成器，只需要生成x，不需要y
    image_name_arr = glob.glob(os.path.join(test_path,"*.tif"))
    total_nb = len(image_name_arr)
    new_img = np.zeros((batch_size,)+image_size)
    batch_count = 0
    for i in range(total_nb):
        img,_ = adjustData(*getData(image_name_arr[i]))
        new_img[batch_count,:,:,:] = img
        batch_count += 1
        if batch_count == batch_size:
            yield new_img
            new_img = np.zeros((batch_size,)+image_size)
            batch_count = 0

# def geneTrainNpy(image_path,mask_path):
    ##将所有数据都存进内存中，小数据集可以使用
    # image_name_arr = glob.glob(os.path.join(image_path,"*.tif"))
    # mask_name_arr = glob.glob(os.path.join(mask_path,"*.tif"))
    # image_arr = []
    # mask_arr = []
    # for i in range(len(image_name_arr)):
        # img,mask = adjustData(*getData(image_name_arr[i],mask_name_arr[i]))
        # image_arr.append(img)
        # mask_arr.append(mask)
    # image_arr = np.array(image_arr)
    # mask_arr = np.array(mask_arr)
    # return image_arr,mask_arr
    
def adjustResult(result, p=0.5):
    #生成的结果是0-1的，实现二分类还需指定阈值，默认0.5，超参数可调
    tmp = result.copy()
    tmp[tmp >= p] = 255
    tmp[tmp < p] = 0
    out = tmp.astype(np.uint8)
    return out
    
def my_accuracy(y_true, y_pred):
    #混淆矩阵中的准确度
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    acc = K.abs(y_true_f - y_pred_f)
    return 1-K.mean(acc)
    
