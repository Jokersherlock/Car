import numpy as np
x = 332.521
x = int(np.floor(x*100))
print(x)
x = format(x, '04X')
Stop=0
Finish=0
Direction=0
message = str(Stop)+str(Finish)+str(Direction)+str(x)
hex_string = format(int(message,16), '07X')
print(message)
print(hex_string)