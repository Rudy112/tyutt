import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from PIL import Image
from numpy.lib.npyio import load
import cv2

path = "/home/wei/catkin_ws/src/realsense-ros/realsense2_camera/scripts/cloth_project/cloth_datasets"


img = cv2.imread('/home/wei/catkin_ws/src/realsense-ros/realsense2_camera/scripts/cloth_project/cloth_datasets/rgb_222_1638890816-279461384.png')
# print(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1)
plt.imshow(img)
# cv2.waitKey(0)        
  
# Destroying present windows on screen
# cv2.destroyAllWindows() 


depth = np.load('/home/wei/catkin_ws/src/realsense-ros/realsense2_camera/scripts/cloth_project/cloth_datasets/depth_222_1638890816-279461384.npy')

plt.subplot(1, 2, 2)
plt.imshow(depth, cmap='gray')
plt.show()


# filename = [f for f in os.listdir(path) if f.startswith("rgb")]
# print(filename)
# filename.sort()
# print(filename)


# valid_images = [".jpg",".gif",".png",".tga"]
# imgs = []
# for f in os.listdir(path):
#     ext = os.path.splitext(f)[1]
#     if ext.lower() not in valid_images:
#         continue
#     imgs.append(Image.open(os.path.join(path,f)))
#     # print(imgs)

# idx = 0
# imidx = imgs[idx].split("_")[1].replace(".png", "")


# # def load_images_from_folder(folder):
# #     images = []
# #     for filename in os.listdir(folder):
# #         img = cv2.imread(os.path.join(folder,filename))
# #         if img is not None:
# #             images.append(img)
# #     return images





# valid_depth = [".npy"]
# depth = []
# for f in os.listdir(path):
#     ext = os.path.splitext(f)[1]
#     if ext.lower() not in valid_depth:
#         continue
#     depth.append(np.load(os.path.join(path,f)))




# rgb_image = imgs[176]
# print(rgb_image.size)


# plt.subplot(1, 2, 1)
# plt.imshow(rgb_image)


# im = np.array(depth[176])
# print(im.shape)
# print(im.dtype)

# plt.subplot(1, 2, 2)
# plt.imshow(im, cmap='gray')
# plt.show()


   

