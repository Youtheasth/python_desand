import cv2
import numpy as np


img = cv2.imread(r"D:\WorkDir\DeSand\Images\DSC_3165.JPG")
# cv2.namedWindow('1',cv2.WINDOW_NORMAL) 
# cv2.imshow('1',img)
# cv2.waitKey(0)

row, col, plane = img.shape
temp = np.zeros((row, col, plane), np.uint8)
temp[:, :, 0] = img[:, :, 0]
cv2.imwrite(r'D:\WorkDir\DeSand\newimg\BlueChannel.JPG', temp)
print(1)
temp = np.zeros((row, col, plane), np.uint8)
temp[:, :, 1] = img[:, :, 1]
cv2.imwrite(r'D:\WorkDir\DeSand\newimg\GreenChannel.JPG', temp)

temp = np.zeros((row, col, plane), np.uint8)
temp[:, :, 2] = img[:, :, 2]
cv2.imwrite(r'D:\WorkDir\DeSand\newimg\RedChannel.JPG', temp)
# temp[:,:,0]=img[:,:,0]
# temp[:,:,1]=img[:,:,1]
# temp[:,:,2]=img[:,:,2]
# cv2.imshow('1', temp)
# cv2.waitKey(0)