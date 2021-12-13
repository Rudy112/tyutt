import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from PIL import Image
from numpy.lib.npyio import load
import cv2
# import rospy
import glob
import pandas as pd
# from visualization_msgs.msg import Marker

path = "E://cloth_datasets//cloth_datasets"
filepath = "E://cloth_datasets//cloth_datasets//clean_datasets"
# img = cv2.imread('E:\\cloth_datasets\\cloth_datasets\\rgb_222_1638896540-155743599.png')

# print(type(cv2.imread(path)))

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.subplot(1,2,1)
# plt.imshow(img)
# depth = np.load('E:\\cloth_datasets\\cloth_datasets\\depth_222_1638896540-155743599.npy')

# plt.subplot(1, 2, 2)
# plt.imshow(depth, cmap='gray')
# plt.show()
class Data_loader():
    
    def __init__(self):
        path = "E://cloth_datasets//cloth_datasets"
        self.path = path
        self.image_list = []
        self.image_index_list = []
        
        self.depth_list = []
        self.depth_index_list = []
        
        for filename in glob.glob('E:/cloth_datasets/cloth_datasets/rgb*'): 
            
            self.rgb_image = Image.open(filename)
            self.image_list.append(self.rgb_image)
            
            self.rgb_image_index = self._get_imidx(filename)
            self.image_index_list.append(self.rgb_image_index)
            
            
        for filename in glob.glob('E:/cloth_datasets/cloth_datasets/depth*'): 
            self.depth_image = np.load(filename)
          
            self.depth_image = self._fill_hole(self.depth_image)
            self.depth_image = Image.fromarray(self.depth_image)
            self.depth_image = self._depth_normalize(self.depth_image)
            self.depth_list.append(self.depth_image)
            
            self.depth_image_index = self._getdep_imidx(filename)
            self.depth_index_list.append(self.depth_image_index)
            
        
    def _depth_normalize(self,depth_data):
        MIN = np.min(depth_data)
        MAX = np.max(depth_data)
        depth_data = (depth_data-MIN) / (MAX-MIN)
        return depth_data
    
    def _fill_hole(self, im):
        zeros = np.where(im == 0)
        mask = np.zeros_like(im, np.uint8)
        mask[zeros] = 1
        im = cv2.inpaint(im, mask, 3, cv2.INPAINT_NS)
        return im
    
    def _get_imidx(self,index):
        imidx = index.split("_")[4].replace(".png", "")
        return imidx
    
    def _getdep_imidx(self,index):
        imidx = index.split("_")[4].replace(".npy", "")
        return imidx
    



   
    
   
       
if __name__ == '__main__':

    a = Data_loader()
    
    rgb_image_list = a.image_list
    rgb_header_list = a.image_index_list
    # rgb_list = list(zip(rgb_header_list, rgb_image_list))
    
    depth_image_list = a.depth_list
    depth_header_list = a.depth_index_list
    # depth_list = list(zip(depth_header_list, depth_image_list))

    align_list = []
    align_image = []
    align_depth = []
    
    for i in range(len(rgb_header_list)):
        rgb_idx = rgb_header_list[i]
        try:
            dep_idx = depth_header_list.index(rgb_idx)
            align_list.append(rgb_idx)
            align_image.append(rgb_image_list[i])
            align_depth.append(depth_image_list[dep_idx])
        except:
            pass
    
    align_data = list(zip(align_list, align_image, align_depth))
    
    # df = pd.DataFrame(align_image, index = align_list)
    for i in range(len(align_data)):
        print(i)
        img_path = os.path.join(filepath, 'rgb_%d_%s-%s.png' % (i, align_data[i][0].split("-")[0], align_data[i][0].split("-")[1]))
        print(align_data[i][1])
        a = np.array(align_data[i][1])
        a = a[:, :, ::-1].copy() # cover to RGB
        cv2.imwrite(img_path, a)        
       
        img_path = os.path.join(filepath, 'depth_%d_%s-%s.npy' % (i, align_data[i][0].split("-")[0], align_data[i][0].split("-")[1]))
        print(align_data[i][2])
        a = np.array(align_data[i][2])
        np.save(img_path, a)
        
        
    plt.subplot(2, 2, 1)
    plt.imshow(align_data[192][1])
 
    
    plt.subplot(2, 2, 2)
    plt.imshow(align_data[192][2], cmap='gray')
    
    plt.subplot(2, 2, 3)
    plt.imshow(align_data[152][1])
 
    plt.subplot(2, 2, 4)
    plt.imshow(align_data[152][2], cmap='gray')

    plt.show()


   

