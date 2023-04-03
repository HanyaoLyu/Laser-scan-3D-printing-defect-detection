import serial
import time
import ImageCaptureZ as ICZ
import cv2
import numpy as np

def Zdet():    
    portx="COM6"
    bps=115200
    timex=5
    ser=serial.Serial(portx,bps,timeout=timex)
    I=22.5
    B=0
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    # cv2.imshow('frame',frame)
    while(True):
        ser.write(str.encode("G01 X30 Y100 F5000 Z%f \r\n"%I))
        ser.readline()
        time.sleep(0)
        D=ICZ.snapshotZ()
        print(D)
        a=D[0]
        b=D[1]
        c=D[2]
        d=D[4]
        if c==0:
            if a==0:
                I=I+0.3
            if d==0:
                I=I+0.1
            else:
                I=I+0.04
        if c>0:
            if a==0:
                I=I-0.3
            if d==0:
                I=I-0.1
            else:
                I=I-0.04
        print(I)
        # print(I,b)
        if b>0:
            D=ICZ.snapshotZ()
            b=D[1]
            if b>0:
                B=1
        if B==1:
            print('The final height is:',I)
            break
    ser.close()
    return I
# Zdet()


portx="COM6"
bps=115200
timex=5
ser=serial.Serial(portx,bps,timeout=timex)
I=23.5
B=0
cap = cv2.VideoCapture(0)
i=0
pp=[]
ser.write(str.encode("G01 X10 Y100 F5000 Z%f \r\n"%I))
while(True):
    i=i+1
    if i==2:
        break
    D=ICZ.snapshotZ()
    P=D[3]
    print([P,i])
    pp.append(P)
# print(np.average(pp),max(pp)-min(pp),np.std(pp))
# print(np.average(pp))
# print('the variance is: %d' % np.var(pp))
# print('the standard divation is: %d'% np.sqrt(pp))





