#road lane detection ...continued

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


img = cv.imread('road4.jpg')

print(img.dtype)

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)#convert to RGB as we want see the output in malplotlib


#defining the ROI
h,w,_ = img.shape
print(h,w)

roi_vertices = [(0,h),(260,25),(w,h)]

#masking the rest of the area
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)# this retuns a black image of the same shape of image given in the argument
    color_count = 255#applying white color. 
    masked = cv.fillPoly(mask,vertices,color_count)# the area under the vertices will be filled by white color
    masked_image = cv.bitwise_and(img,masked) # finding intersection between image and masked image in order to find the region of interst
    return masked_image

#draw lines
def draw_lines(img,lines):
    img = np.copy(img)
    black_image = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(black_image,(x1,y1),(x2,y2),(0,0,255),3)
            
        img = cv.addWeighted(img, 0.8, black_image, 1,0.0)
        return img
    

img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(img_gray,(5,5),cv.BORDER_DEFAULT)
canny_edge = cv.Canny(blur,100,200,apertureSize=3)

#using the defined function
cropped_image= region_of_interest(canny_edge, np.array([roi_vertices],np.int32))



lines = cv.HoughLinesP(cropped_image, 6, np.pi/180, 100, minLineLength=100, maxLineGap=500)


image_with_lines = draw_lines(img,lines)


cv.imshow('img',image_with_lines)
cv.imshow('img',image_with_lines)
cv.imshow('img1',cropped_image)
cv.imshow('img2',canny_edge)



                                  
cv.waitKey(0)
cv.destroyAllWindows()                                  
                                  
#show the image in matplotlib
#plt.subplot(121)
#plt.imshow(image_with_lines)
#plt.subplot(122)
#plt.imshow(cropped_image)
#plt.show()
