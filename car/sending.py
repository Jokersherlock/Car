import serial
def send_message(x,Stop=0,Finish=0,Direction=0):
    ser = serial.Serial('COM1',38400)
    '''
    message = str(Stop)+str(Finish)+(Direction)+str(x)
    message = format(int(message,16), '07X')
    ser.write(message)
    ser.close()
    '''
    Stop = Stop.to_bytes(2,byteorder='big')
    Finish = Finish.to_bytes(2,byteorder='big')
    x = x.to_bytes(2,byteorder='big')
    Direction = Direction.to_bytes(2,byteorder='big')
    message = b'\x28'+b'\x28'+Stop+Finish+Direction+x+b'\x29'
    # 发送信息
    ser.write(message)
    # 关闭串口
    ser.close()