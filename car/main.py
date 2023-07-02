from tool import gen_model
from tool import img_tran
from tool import outputhandling
import numpy as np
import tracing
import cv2
from sending import send_message
model,LABEL_NAMES = gen_model()
cap = cv2.VideoCapture(1)
cv2.namedWindow("img")
while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from camera")
            break
        '''
        kernel_size = (5,5)
        sigma = 0
        frame = cv2.GaussianBlur(frame,kernel_size,sigma)
        img = img_tran(frame)
        preds = model(img)
        output,img= outputhandling(preds,frame,draw=True)
        cv2.imshow("img",frame)
        '''
        Stop = tracing.detect_white_line(frame)
        
        '''
        if Stop:
            text = "True"
        else:
             text = "False"
        
        position = (50, 50)  # 文本位置的坐标，左上角为原点

        # 设置字体、字号、颜色和线宽
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)  # 文本颜色，BGR格式
        thickness = 2  # 文本线宽

        # 在图像上写入文本
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
        '''

        x = tracing.tracing(frame)
        height, width, _ = frame.shape
        x = int(np.floor(x*100))
        #x = format(x, '04X')
        send_message(x)
        '''
        position=(50,100)
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
        cv2.imshow("frame",frame)
        '''
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()