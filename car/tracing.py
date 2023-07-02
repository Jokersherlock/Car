import cv2
import numpy as np

def dilate_image(image, kernel_size, iterations):
    # 创建结构元素（膨胀核）
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 应用膨胀操作
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)

    return dilated_image

def erode_image(image, kernel_size, iterations):
    # 创建结构元素（腐蚀核）
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 应用腐蚀操作
    eroded_image = cv2.erode(image, kernel, iterations=iterations)

    return eroded_image

def convert_to_binary(image):
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 对灰度图像进行二值化处理
    _, binary_image = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)

    # 显示原始图像和处理后的图像
    #cv2.imshow("Binary Image", binary_image)
    return binary_image

def get_red(image):
    # 加载图像
    # 转换颜色空间：从BGR到HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色范围
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # 创建掩码
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # 应用掩码
    red_extracted = cv2.bitwise_and(image, image, mask=red_mask)

    #print(red_extracted)
    # 显示结果
    #cv2.imshow("Original Image", image)
    #cv2.imshow("Red Extracted Image", red_extracted)
    return red_extracted

def tracing(img):
    height, width, _ = img.shape
    x_start, x_end = 0, width   # 列范围
    y_start, y_end = np.floor(height/2+100), height   # 行范围

    # 对图像进行切片操作
    img = img[int(y_start):int(y_end),int(x_start):int(x_end)]
    kernel_size = (5,5)
    sigma = 0
    img = cv2.GaussianBlur(img,kernel_size,sigma)
    dst = get_red(img)
    dst = convert_to_binary(dst)
    dst = dilate_image(dst, kernel_size=5, iterations=1)
    height, width, _ = img.shape
    cnt = 0
    x = 0
    for i in range(0,height):
        for j in range(0,width):
            if(not dst[i][j]):
                cnt = cnt + 1
                x = x + j
    x = x/cnt
    '''
    image_with_line = img.copy()

    # 在图像副本上绘制线条

    start_point = (int(np.floor(x)),0)  # 使用int函数转换浮点数为整数
    end_point = (int(np.floor(x)),height-1)
    color = (0, 255, 0)   # 在OpenCV中，颜色通道顺序为BGR
    thickness = 2   # 线条的粗细

    cv2.line(image_with_line, start_point, end_point, color, thickness)
    # 显示绘制了线条的图像
    
    start_point = (int(width/2),int(height/2))  # 使用int函数转换浮点数为整数
    end_point = (int(width/2),height-1)
    color = (0, 0, 255)   # 在OpenCV中，颜色通道顺序为BGR
    thickness = 1   # 线条的粗细
    cv2.line(image_with_line, start_point, end_point, color, thickness)
    cv2.imshow('Image with Line', image_with_line)
    print(x-width/2)
    '''
    return x

def detect_white_line(img, threshold=200, line_length=50):
    # 读取图像并转换为灰度图像
    
    height, width, _ = img.shape
    x_start, x_end = 0, width   # 列范围
    y_start, y_end = np.floor(height/2+50),  np.floor(height/2+100)   # 行范围

    # 对图像进行切片操作
    img = img[int(y_start):int(y_end),int(x_start):int(x_end)]
    
    kernel_size = (5,5)
    sigma = 0
    img = cv2.GaussianBlur(img,kernel_size,sigma)
    dst = get_red(img)
    dst = convert_to_binary(dst)
    dst = dilate_image(dst, kernel_size=5, iterations=1)
    #cv2.imshow("dst",dst)
    height, width, _ = img.shape
    for i in range(0,height):
        for j in range(0,width):
            if(dst[i][j]):
                dst[i][j] = 1
    # 水平投影
    projection = np.sum(dst, axis=0)
    cnt = 0
    for i in range(width):
        if  projection[i]:
            cnt = cnt + 1
    if cnt > 200:
        return True
    return False








'''
if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    cv2.namedWindow("img")
    while True:
            # 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame from camera")
                break
            
            x = tracing(frame)
            cv2.imshow("frame",frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

'''