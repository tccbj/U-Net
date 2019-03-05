# coding:utf8
from osgeo import gdal
import os
import glob
import numpy as np

#读图像文件
def read_img(filename):
    dataset = gdal.Open(filename)       #打开文件

    im_width = dataset.RasterXSize    #栅格矩阵的列数
    im_height = dataset.RasterYSize   #栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform()  #仿射矩阵
    im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵

    del dataset 
    return im_proj,im_geotrans,im_data

#写文件，以写成tif为例
def write_img(filename,im_proj,im_geotrans,im_data):
    #gdal数据类型包括
    #gdal.GDT_Byte, 
    #gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    #gdal.GDT_Float32, gdal.GDT_Float64

    #判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    #判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape 

    #创建文件
    driver = gdal.GetDriverByName("GTiff")            #数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)              #写入仿射变换参数
    dataset.SetProjection(im_proj)                    #写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data[0])  #写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset

def prj_exist(image_path, save_path):
    image_name_arr = glob.glob(os.path.join(image_path,"*.tif"))
    for i in range(len(image_name_arr)):
        out_name = image_name_arr[i].replace(image_path,save_path)
        out_name = out_name.replace('.tif','_p.tif')
        proj,geotrans,_ = read_img(image_name_arr[i])        #读数据
        _,_,data = read_img(out_name)
        #print(proj)
        #print(geotrans)
        #print(data)
        #print(data.shape)
        out_path = out_name.replace('.tif','_pp.tif')
        write_img(out_path,proj,geotrans,data) #写数据

def saveResultPrj(in_path,save_path,result):
    #将预测结果保存成图片
    #io.imsave为png时需要三波段，tif可以只有一波段
    image_name_arr = glob.glob(os.path.join(in_path,"*.tif"))
    for i,item in enumerate(result):
        out_name = image_name_arr[i].replace(in_path,save_path)
        out_name = out_name.replace('.tif','_pp.tif')
        proj,geotrans,_ = read_img(image_name_arr[i])        #读数据
        item = item.transpose(2,0,1)
        write_img(out_name,proj,geotrans,item)
    
def rotateData(infile, outfile):
    proj,geotrans,data = read_img(infile)        #读数据
    new_data = np.zeros_like(data)
    if len(data.shape) == 3:
        im_bands, im_height, im_width = data.shape
    else:
        im_bands, (im_height, im_width) = 1,data.shape 
    for i in range(im_bands):
        new_data[i] = data[i].T
    
    write_img(outfile,proj,geotrans,new_data)
    
if __name__ == "__main__":
    image_path = r'E:\tree\randomdata2\train\image3'
    image_name_arr = glob.glob(os.path.join(image_path,"*.tif"))
    print(len(image_name_arr))
    for i in range(len(image_name_arr)):
        rotateData(image_name_arr[i],image_name_arr[i].replace('.tif','_r.tif'))
