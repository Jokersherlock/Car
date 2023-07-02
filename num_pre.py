def get_direction(model,target,img):
    target = str(target)
    prev = model.detect(img)
    cnt = len(prev)
    locations=[]
    nums = []
    flag = False
    for i in range(0,cnt):
        location,num,p=prev[i]
        if num == target:
            flag = True
        locations.append(location)
        nums.append(num)
    if flag:
        index = nums.index(target)
        Sorted_location = sorted(locations, key=lambda x: x[0])
        nindex = Sorted_location.index(locations[index])
        if nindex<=1:
            return 1
        else:
            return 2
    else:
        return 0
    
def get_num(model,img):
    prev = model.detect(img)
    cnt = len(prev)
    P=[]
    nums = []
    flag = False
    for i in range(0,cnt):
        location,num,p=prev[i]
        P.append(p)
        nums.append(num)
    Sorted_p = sorted(P)
    num = nums[P.index(Sorted_p[-1])]
    return num


