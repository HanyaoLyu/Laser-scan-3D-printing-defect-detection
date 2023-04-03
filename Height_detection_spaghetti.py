# image processing based on laser stripe position
# Edited by Hanyao Lyu

import cv2
import numpy as np
import matplotlib.pyplot as plt
def Height(filename):
    cv2.getBuildInformation()
    img = cv2.imread(filename)
    #img=cv2.transpose(img)
    img_blur = cv2.GaussianBlur(img, (3,3), 0) 
    denoise = cv2.fastNlMeansDenoisingColored(img_blur,None,20,15,7,21)
    # path1="C:/out/Spaghetti detection result/cropped.jpg"
    # path2="C:/out/Spaghetti detection result/imageplot.jpg"
    # path3="C:/out/Spaghetti detection result/binarized.jpg"
    #determine whether image is color 
    # if(len(img.shape)<3):
    #     print ('gray')
    # elif len(img.shape)==3:
    #     print ('Color(RGB)')
    # else:
    #     print ('others')

    grid_RGB = cv2.cvtColor(denoise, cv2.COLOR_BGR2RGB)

    # 从RGB色彩空间转换到HSV色彩空间
    grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)

    # H、S、V范围一：
    lower1 = np.array([0,43,46])
    upper1 = np.array([10,255,255])
    mask1 = cv2.inRange(grid_HSV, lower1, upper1)       # mask1 为二值图像
    # res1 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask1)

    # H、S、V范围二：
    lower2 = np.array([156,43,46])
    upper2 = np.array([180,255,255])
    mask2 = cv2.inRange(grid_HSV, lower2, upper2)
    # res2 = cv2.bitwise_and(grid_RGB,grid_RGB, mask=mask2)

    # 将两个二值图像结果 相加
    mask3 = mask1 + mask2

    # 结果显示

    # cv2.imshow("mask3", mask3)

    # plot original image
    # cv2.imshow('Original', img)
    # click any buttons to continue
    # cv2.waitKey(0)
    # cropped image to range of interesting
    hgt, wdt = mask3.shape[:2]
    start_row, start_col = int(hgt * .15), int(wdt * .25)
    end_row, end_col = int(hgt * .85), int(wdt * .75)
    cropped = mask3[start_row:end_row , start_col:end_col]



    hgt,wdt=cropped.shape[:2]
    # print(hgt,wdt)

    start_row, start_col = int(hgt * 0.4), int(wdt/2-2)
    end_row, end_col = int(hgt * 0.60), int(wdt/2+2)
    cropped1 = cropped[start_row:end_row , start_col:end_col]
    hgt,wdt=cropped1.shape[:2]

    N=0

    sumj=0
    ## find all the non-zero points
    for i in range(0,wdt):
        for j in range(0,hgt):
            if cropped1[j][i]!=0:
                sumj=sumj+j
                N=N+1
    if N==0:
        P=0      
    if N!=0:
        P=(sumj/N)
    print("The laser stripe average middle position is",P)
    if P < hgt/2-8 or P > hgt/2+8:
        A = 0
    else:
        A = 1

    if P>hgt/2-2 and P<hgt/2+2:
        B=1
    else:
        B=0

    if P>hgt/2:
        C=1
    else:
        C=0

    if P>hgt/2-5 and P<hgt/2+5:
        D=1
    else:
        D=0

    print(A)
    # imageplot = plt.imshow(edges)
    # plt.title ('edges')
    # plt.xlabel('pixel width')
    # plt.ylabel('pixel length')
    # # plt.show()

    # img=np.array(img)
    # cropped = np.asarray(cropped) #将变量类型转换为numpy数组
    # edges=np.asarray(edges)
    # plt.subplot(131), plt.imshow(img), plt.title('Original')
    # plt.subplot(132), plt.imshow(mask3), plt.title('Binarized')
    # # # # # # plt.axis('off')
    # # # # print(hgt)

    # plt.subplot(133), plt.imshow(cropped1, 'gray'), plt.title('ROI:mid position: %d'%P)
    # # # plt.axis('off')
    # plt.show()
    # cv2.waitkey(0)
    A=(A,P,B,C,D)
    return A
# Height("C:/out/1.jpg")