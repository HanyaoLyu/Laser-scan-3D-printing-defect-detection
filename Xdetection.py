

import serial
import time
import ImageCaptureX as ICX
# import Zdet 
def Xdet():    
    portx="COM5"
    bps=115200
    timex=5
    ser=serial.Serial(portx,bps,timeout=timex)
    I=-33.5 # initial guess or not sure distance between sensor and nozzle
    i=0
    Di=0
    B=0
    # Z=Zdet.Zdet()+0.2
    while(True):
        x=44.8+I # the coordinate for the nozzle 
        ser.write(str.encode("G01 X%f Y85 Z22.5 \r\n"%(x))) #go to that position
        ser.readline() 
        time.sleep(1)# wait
        D=ICX.snapshotX()
        print(abs(Di-D))
        if i==0:
            C=1
        if abs(Di-D)<1.5 or C==1: # the edge was not detected, keep working
            print('Keep calibrating') 
            I=I-0.1
        if abs(Di-D)>1.5 and C!=1: # the edge was detected 
            print('Shinning')
            D=ICX.snapshotX() 
            if abs(Di-D)>1.5 and C!=1: # check again
                print('shininging')
                B=1
                break
        if B==1:
            print('finish')
            break ##if detected, jump out and feedback the distance
        C=0


        i=i+1
        Di=D
    I=I-1.6 
    print(I)
    return I

# Xdet()