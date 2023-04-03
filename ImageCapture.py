
import cv2
import Height_detection_spaghetti as HS
import time
cap = cv2.VideoCapture(1)

def snapshot():


    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()   
        
        cv2.imwrite("C:/out/1.jpg",frame)
        time.sleep(0.1)
        break
        # if cv2.waitKey(0):
        #     break
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


    # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows() 
    # cv2.imshow('frame',frame)
    P=HS.Height("C:/out/1.jpg")
    B=P[1]
    print(B)
    A=P[0]
    p=P[1]
    b=P[2]
    C=P[3]
    D=P[4]
    A=(A,b,C,p,D)
    return A
# (A,b,C,p)=snapshot()
# print(p)