import numpy as np
import tracing
import cv2
from sending import send_message
import Vision_Net
from num_pre import *
model = Vision_Net.FastestDet()
cap = cv2.VideoCapture(0)
#cv2.namedWindow("img")
target = 0
mode = 1
END_TIMES = 5
while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera")
            break
        if mode == 0:
             target = get_num(model,frame)
             mode == 1
        x = tracing.tracing(frame)
        #print(x)
        height, width, _ = frame.shape
        x = x - width/2
        #print(x)
        x = x + 100
        if x>200:
           x=200
        if x<0:
           x=0
        x = int(np.floor(x*100))
        x = x+256
        if mode == 1:
            cnt = 0
            direction = 0
            Stop = tracing.detect_white_line(frame)
            if(Stop):
                mode = 2
                send_message(x,Stop=1)
            else:
                send_message(x)
        if mode == 2:
            temp_direction = get_direction(model,target,frame)
            direction = direction*cnt/(cnt+1)+temp_direction/(cnt+1)
            cnt = cnt + 1
            if(cnt >= END_TIMES):
                cnt = 0
                send_message(x,Finish=1,direction=int(round(direction)))
                send_message(x,Finish=1,direction=int(round(direction)))
                send_message(x,Finish=1,direction=int(round(direction)))
                send_message(x,Finish=1,direction=int(round(direction)))
                send_message(x,Finish=1,direction=int(round(direction)))
                direction = 0
                mode = 1
            else:
                send_message(x,direction=int(round(direction)))

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()