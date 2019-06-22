import numpy as np
workhouse_dict = {'CRH2': [1, 2, 1.5, 4, 7],      # CRH2:
                  'CRH3': [0.8, 2.4, 0.5, 4.8, 6.5],  # CRH3:
                  'CRH5': [1.3, 2.5, 1.5, 3, 6.5],  # CRH5:
                  'CRH6': [1, 2.7, 0.3, 5, 7]}
arrive_time_tb = np.array([[960, 'CRH2', 'IV'],
                           [2820, 'CRH5', 'II'],
                           [4920, 'CRH2', 'II'],
                           [7200, 'CRH6', 'I'],
                           [8460, 'CRH3', 'III'],
                           [10920, 'CRH6', 'II'],
                           [12660, 'CRH2', 'V'],
                           ])

# 构建表
#      a      b      c      d      e
# I    b      None   None   None   None
# II   b      c      None   None   None
# III  b      d      None   None   None
# IV   c      None   d      e      None
# V    b      c      d      e      None
fix_order = np.array([['b', None, None, None, None],
                      ['b', 'c' , None, None, None],
                      ['b', 'd' , None, None, None],
                      ['c', None, 'd' , 'e' , None],
                      ['b', 'c' , 'd' , 'e' , None]])

str2num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4,
           'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4}

arrive_time_list = arrive_time_tb[:, 0].tolist()
num_cars = arrive_time_tb.shape[0]
workhouse_a = np.zeros((3, 2))
workhouse_b = np.zeros((8, 2))
workhouse_c = np.zeros((5, 2))
workhouse_d = np.zeros((3, 2))
workhouse_e = np.zeros((2, 2))
precar_a = []
precar_b = []
precar_c = []
precar_d = []
precar_e = []
finish = []

def check_next_workhouse(car_num, wh_type):
    idx = int(car_num) - 1
    fix_type = arrive_time_tb[idx, 2]  # 'IV'
    wk_num = str2num[wh_type]
    fix_num = str2num[fix_type]
    next_wk = fix_order[fix_num, wk_num] # 'a', None
    if next_wk == 'b':
        precar_b.append(car_num)
    if next_wk == 'c':
        precar_c.append(car_num)
    if next_wk == 'd':
        precar_d.append(car_num)
    if next_wk == 'e':
        precar_e.append(car_num)
    if next_wk is None:
        finish.append(car_num)
        print('yes')



car_num = 0
for i in range(50000):
    if str(i) in arrive_time_list:
        car_num += 1
        print(car_num)
        precar_a.append(car_num)

    for j in workhouse_a:
        if j[0] != 0:  # 如果a车间该车位被占有
            j[1] += 1  # 该位置的车辆时间增加1

            car_type = arrive_time_tb[int(j[0]) - 1, 1]  # 计算该车的类型
            car_params = workhouse_dict[car_type]  # 查询字典返回该车的系数

            if j[1] == 3600 * car_params[0]:  # 如果该位置车时间到达目标

                check_next_workhouse(j[0], 'a')  # 函数将车加入到正确的等待区

                j[0] = 0  # 初始该位置车号
                j[1] = 0  # 初始该位置时间
        if j[0] == 0:  # 如果a车间该车位为空
            if len(precar_a) > 0:  # 等待a区有车才送入a车间
                j[0] = precar_a[0]
                precar_a.remove(precar_a[0])
            else:
                pass

    for j in workhouse_b:
        if j[0] != 0:  # 如果a车间该车位被占有
            j[1] += 1  # 该位置的车辆时间增加1

            car_type = arrive_time_tb[int(j[0]) - 1, 1]  # 计算该车的类型
            car_params = workhouse_dict[car_type]  # 查询字典返回该车的系数

            if j[1] == 3600 * car_params[1]:  # 如果该位置车时间到达目标

                check_next_workhouse(j[0], 'b')

                j[0] = 0  # 初始该位置车号
                j[1] = 0  # 初始该位置时间
        if j[0] == 0:  # 如果a车间该车位为空
            if len(precar_b) > 0:  # 等待a区有车才送入a车间
                j[0] = precar_b[0]
                precar_b.remove(precar_b[0])
            else:
                pass

    for j in workhouse_c:
        if j[0] != 0:  # 如果a车间该车位被占有
            j[1] += 1  # 该位置的车辆时间增加1

            car_type = arrive_time_tb[int(j[0]) - 1, 1]  # 计算该车的类型
            car_params = workhouse_dict[car_type]  # 查询字典返回该车的系数

            if j[1] == 3600 * car_params[2]:  # 如果该位置车时间到达目标

                check_next_workhouse(j[0], 'c')

                j[0] = 0  # 初始该位置车号
                j[1] = 0  # 初始该位置时间
        if j[0] == 0:  # 如果a车间该车位为空
            if len(precar_c) > 0:  # 等待a区有车才送入a车间
                j[0] = precar_c[0]
                precar_c.remove(precar_c[0])
            else:
                pass

    for j in workhouse_d:
        if j[0] != 0:  # 如果a车间该车位被占有
            j[1] += 1  # 该位置的车辆时间增加1

            car_type = arrive_time_tb[int(j[0]) - 1, 1]  # 计算该车的类型
            car_params = workhouse_dict[car_type]  # 查询字典返回该车的系数

            if j[1] == 3600 * car_params[2]:  # 如果该位置车时间到达目标

                check_next_workhouse(j[0], 'd')

                j[0] = 0  # 初始该位置车号
                j[1] = 0  # 初始该位置时间
        if j[0] == 0:  # 如果a车间该车位为空
            if len(precar_d) > 0:  # 等待a区有车才送入a车间
                j[0] = precar_d[0]
                precar_d.remove(precar_d[0])
            else:
                pass

    for j in workhouse_e:
        if j[0] != 0:  # 如果a车间该车位被占有
            j[1] += 1  # 该位置的车辆时间增加1

            car_type = arrive_time_tb[int(j[0]) - 1, 1]  # 计算该车的类型
            car_params = workhouse_dict[car_type]  # 查询字典返回该车的系数

            if j[1] == 3600 * car_params[2]:  # 如果该位置车时间到达目标

                check_next_workhouse(j[0], 'e')

                j[0] = 0  # 初始该位置车号
                j[1] = 0  # 初始该位置时间
        if j[0] == 0:  # 如果a车间该车位为空
            if len(precar_e) > 0:  # 等待a区有车才送入a车间
                j[0] = precar_e[0]
                precar_e.remove(precar_e[0])
            else:
                pass
    if len(finish) == num_cars:
        print(i)
        break
