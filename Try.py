# import numpy as np
import os
import cv2
import ImageCapture as IC
import serial
import time
import gcodeparser as gcp
import Zdet 
import Xdetection as Xdet
import ImageCaptureZ as ICZ
# detect the z distance
Z=Zdet.Zdet() 
# find x coordinate
a=Xdet.Xdet()
# find y coordinate
portx="COM5"#find the correct port
bps=115200
timex=5
ser=serial.Serial(portx,bps,timeout=timex)
# indicate number
I=49.5
N=0
n=0
P1=0
A=0
L=0
while(True):
    Y=44.808+I
    # print(Y)
    ser.write(str.encode("G01 X0 Y%f Z%f \r\n"%(Y,Z)))
    ser.readline()
    time.sleep(1)
    P=ICZ.snapshotZ()
    B=P[1]
    C=P[2]
    D=P[3]
    P=P[5]


    if D==0:
        break
    print(P1,D,A)
    if abs(D-P1)>1:
        A=A+1
        if A>2:
            L=1
    if L==1:
        break
    
    

    if B==1:
        n=1
    # print([P,n,C])

    if P==0 and n==1 and C==0:
        P=ICZ.snapshotZ()
        P=P[5]
        
        if P==0:
            N=1

    if N==1:
        break
    I=I-0.1
    P1=D
b=I
b=b-1.2
print(I)
# find the sampling points
model = gcp.parse_gcode(os.path.abspath("zy322.gcode"), 5, 10, "Min-max")
cap = cv2.VideoCapture(1)
j=50
Height=0.2*j
print(Height)
layer = model.layers[j]
layers=model.layers
layer.to_svg(model.max_y, model.max_x, 'test.svg')

layer.plot_layer('k', 'b')
A=layer.gen_sample_points("Min-max",layers,10,0,2)
B=A[0]
C=A[1]
X=[]
Y=[]
w=[]
for i in B:
    while(True):

        if i>max(B)-0.2:
            i=i-0.3
            X.append(i)
            break
        if i>max(B)-3:
            w.append(i)
            print(w)

        if i<min(B)+0.2:
            i=i+0.3
            X.append(i)
            break
        else:
            X.append(i)
            break
    if len(X)==20:
        break
wn=len(w)
for j in C:
    while(True):
        if j>max(C)-0.2:
            j=j-0.3
            Y.append(j)
            break
        if j<min(C)+0.2:
            j=j+0.3
            Y.append(j)
            break
        else:
            Y.append(j)
            break
    if len(Y)==20:
        break
print(B)
print(X)
print(C)
print(Y)
#IC.snapshot()

#SP.Spaghetti()
#HS.Height()
## relative height for X and Y direction

    
# Point 1
Z=Z-0.65
result=ser.write(str.encode("G01 X%f Y%f Z%f \r\n"% (X[0]+a,Y[0]+b,Height+Z)))
time.sleep(3)
P=IC.snapshot()
P=P[0]
R=0
W=0
if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

#Point 2

result=ser.write(str.encode("G01 X%f Y%f \r\n"% (X[1]+a,Y[1]+b)))
ser.readline()
while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")

time.sleep(1)
P=IC.snapshot()
P=P[0]
if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

#Point3
result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[2]+a,Y[2]+b)))
ser.readline()
# while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
#     print("working...")
# print("movement finished!")

time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

# Point 4
result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[3]+a,Y[3]+b)))

ser.readline()
while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]
if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

# Point 5
result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[4]+a,Y[4]+b)))

ser.readline()
while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

# Point 6
result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[5]+a,Y[5]+b)))

while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

# Point 7
result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[6]+a,Y[6]+b)))

while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

# Point 8
result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[7]+a,Y[7]+b)))

while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

# Point 9
result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[8]+a,Y[8]+b)))
while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

# Point 10
result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[9]+a,Y[9]+b)))

while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

# Point 11
result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[10]+a,Y[10]+b)))

while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

# Point 12
result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[11]+a,Y[11]+b)))

while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

# Point 13
result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[12]+a,Y[12]+b)))

while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

# Point 14

result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[13]+a,Y[13]+b)))

while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

# Point 15
result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[14]+a,Y[14]+b)))

while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

# Point 16
result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[15]+a,Y[15]+b)))

while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

# Point 17
result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[16]+a,Y[16]+b)))

while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)
# Point 18

result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[17]+a,Y[17]+b)))

while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)

# Point 19
result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[18]+a,Y[18]+b)))

while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)


# Point 20
result=ser.write(str.encode("G01 X%f Y%f \r\n" % (X[19]+a,Y[19]+b)))

while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
time.sleep(1)
P=IC.snapshot()
P=P[0]

if P == 1:
    print('Goes well')
    R=R+1
else:
    result=ser.write(str.encode("M300 S440 P200\r\n"))
    print('Goes wrong')
    W=W+1
cv2.waitKey(0)
print(R,W,wn)
print(R+W-(W-wn),W-wn)
result=ser.write(str.encode("M400\r\n"))
ser.readline()
while(ser.readline().decode('utf-8').__contains__('echo:busy: processing\n')):
    print("working...")
print("movement finished!")
result=ser.write(str.encode("M300 S440 P200\r\n"))


ser.close()