import serial

def send_message(x,Stop=0,Finish=0,Direction=0):
    ser = serial.Serial('/dev/ttyAMA0',38400)
    #print(x)
    y = Stop*100 + Finish*10 +Direction
    y = y + 256
    y = y.to_bytes(2,byteorder='big')
    x = x.to_bytes(2,byteorder='big')
    message = b'\x28'+b'\x28'+y+x+b'\x29'
    #print(message)
    # 发送信息
    ser.write(message)
    # 关闭串口
    ser.close()

    

