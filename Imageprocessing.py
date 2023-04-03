# image processing based on laser stripe position
# Edited by Hanyao Lyu

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
def image_process():
      cv2.getBuildInformation()
      img = cv2.imread("C:/out/4.jpg")
      path1="C:/out/Spaghetti detection result/cropped.jpg"
      path2="C:/out/Spaghetti detection result/imageplot.jpg"
      path3="C:/out/Spaghetti detection result/binarized.jpg"
      # alpha=0.5
      # beta=1
      # img1 = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
      # # plt.subplot(1,2,1); plt.imshow(img)
      # # plt.subplot(1,2,2); plt.imshow(img1)
      # # plt.show()
      #determine whether image is color 
      # if(len(img.shape)<3):
      #       print ('gray')
      # elif len(img.shape)==3:
      #       print ('Color(RGB)')
      # else:
      #       print ('others')

      # rows=2;
      # columns=2;
      # plot original image
      # cv2.imshow('Original', img)
      # click any buttons to continue
      # cv2.waitKey(0)
      # cropped image to range of interesting

      hgt, wdt = img.shape[:2]
      start_row, start_col = int(hgt * .25), int(wdt/2-10)
      end_row, end_col = int(hgt * .75), int(wdt/2+10)
      cropped = img[start_row:end_row , start_col:end_col]
      # plot cropped image
      # cv2.imshow('cropped',cropped)
      # cropped=img
      cv2.imwrite('path/cropped.png',cropped)
      cv2.imwrite(path1,cropped)


      hgt,wdt=cropped.shape[:2]
      # print(hgt,wdt)
      # smooth image to reduce unwanted noise point
      #cropped=img
      denoise = cv2.fastNlMeansDenoisingColored(cropped,None,20,15,7,21)
      # plot denoised image 

      # cv2.imshow('cropped image', cropped)
      # cv2.imshow('denoised image', denoise)
      cv2.imwrite('denoise image.png',denoise)
      img_gray = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)
      # Blur the image for better edge detection
      img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
      # cv2.imshow('Blur',img_blur)
      # cv2.imshow('gray scale image',img_gray)
      cv2.imwrite('gray_scale_image.png',img_gray)
      # cv2.imwrite('blured image.png',img_blur)
      ret, otsu = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      # print("Obtained threshold: ", ret)
      # cv2.imshow('binarized',otsu)
      ret,im_bw = cv2.threshold(img_blur,ret,255,0)
      cv2.imwrite(path3,otsu)
      # print(ret)
      # Set lower threshold, the points in the range of ret-100 to ret will be reserve based on their relation to the high light point (>ret), 
      edges = cv2.Canny(im_bw,ret,255) # Canny Edge Detectio
      # Display Canny Edge Detection Image
      # [row_indexes, col_indexes] = np.nonzero(edges)
      # print(row_indexes[1],row_indexes[-1])
      edges=np.asarray(edges)
      # plt.imshow(edges)
      # plt.show()
      I=1
      J=1
      N=0
      J1=0
      J2=0
      P=[]
      W=[]
      for i in range(0,wdt):# set i as increment number for pic width
            for j in range(0,hgt): # set j as increment number for pic height
                  if edges[j][i] != 0: # find the non zero point in edge 找到二值化图像种的非零点（亮点）
                        print(i,I,j)
                        if i==0:# first column detection考虑到之后使用的（差异过大的点=前一个点的坐标）的误差排除方法，对于第一列检测
                              if I == i:
                                    J2=j
                              if I !=i:
                                    J1=j 
                        if I == i:# find highier edge point 找到激光的上边线
                              if abs(J2-j)>3: # whether the coordinates difference between two continuous high edge points larger than 3 点坐标是否和前一列的上边线有较大区别
                                    j=J2 # if yes, large change happened, set this highier edge point equal to the previouos high edge point 随动
                              J2=j # record high edge j coordinate in J2 将这个上边线的点的坐标记录在J2中以便下一轮循环比较 
                              if j-J>10: # width (highier edge - lower edge) should be larger than 5 pixels 检测是否在一个横坐标上出现多个点，如果是，开始运算
                                    P.append((j+J)/2) # middle position coordinates
                                    W.append(j-J)# Laser line width
                        if I != i: # find lower edge point 找到激光的下边缘
                              if abs(j-J1)>3: # whether the height coordinates difference between continuous two lower edge points larger than 3 
                                    j=J1  # if yes, large change happened, set this lower edge point equal to the previouos lower edge point
                              J1=j # record lower edge height coordinate j in J1
                        I=i # record the width coordinate whatever the High or Low edge
                        J=j # record the height coordinate whatever the high or low edge
      if len(P)!=0:
            P=np.mean(P)    
            print("The laser stripe average middle position is",P)   
      else:
          P=0
      # imageplot = plt.imshow(edges)
      # plt.title ('edges')
      # plt.xlabel('pixel width')
      # plt.ylabel('pixel length')
      # plt.show()
      img=np.array(img)
      cropped = np.asarray(cropped) #将变量类型转换为numpy数组
      edges=np.asarray(edges)
      otsu=np.asarray(otsu)
      plt.subplot(141), plt.imshow(img, 'gray'), plt.title('original')
      plt.axis('on')
      plt.subplot(142), plt.imshow(cropped, 'gray'), plt.title('Cropped')
      plt.axis('on')
      plt.subplot(143), plt.imshow(otsu, 'gray'), plt.title('binarized')
      plt.axis('on')
      plt.subplot(144), plt.imshow(edges, 'gray'), plt.title('edges,mid position: %d'%P)
      plt.axis('on')
      plt.show()
      if P < hgt/2-10 or P > hgt/2+10:
        A = 0
      else:
        A = 1

      if P < hgt/2-5 and P > hgt/2-10 or P > hgt/2+5 and P < hgt/2+10:
        D = 0
      else:
        D = 1

      if P>hgt/2-1 and P<hgt/2+1:
        B=1
      else:
        B=0

      if P>hgt/2:
        C=1
      else:
        C=0

      if P>hgt/2-2 and P<hgt/2+2:
        E=1
      else:
        E=0



      # print(A)
      A=(A,P,B,C,D,E)
      return A
image_process()