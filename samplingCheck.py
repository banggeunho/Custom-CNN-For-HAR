import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# 경로 안에 있는 모든 파일 읽어온다.
# 여기 경로 수정해서 사용하세용~
path = './download/'
current_path = os.getcwd()

# 폴더를 만들어줍니다.
for i in range(1, 14):
    os.makedirs(current_path + "/time/"+str(i), exist_ok = True)

length = 0
file_len = len(os.listdir(path))
complete_file_len = 0

# coding=utf-8
def is_number(num):
    try:
        float(num)
        return True  # num을 float으로 변환할 수 있는 경우
    except ValueError:  # num을 float으로 변환할 수 없는 경우
        return False


for filename in os.listdir(path):

    heartR = pd.DataFrame(columns=['level', 'time', 'heartR'])
    accelX = pd.DataFrame(columns=['level', 'time', 'accelX'])
    accelY = pd.DataFrame(columns=['level', 'time', 'accelY'])
    accelZ = pd.DataFrame(columns=['level', 'time', 'accelZ'])
    gyroX = pd.DataFrame(columns=['level', 'time', 'gyroX'])
    gyroY = pd.DataFrame(columns=['level', 'time', 'gyroY'])
    gyroZ = pd.DataFrame(columns=['level', 'time', 'gyroZ'])
    stepC = pd.DataFrame(columns=['level', 'time', 'stepC'])

    with open(path+filename,  encoding='utf-8') as f:
        lines = f.read()
        filename = filename[:-4]
        print(f'[{complete_file_len}/{file_len}, {complete_file_len/file_len * 100:.2f}%]{filename}')
        # 라인별로 ', '(콤마 공백) 로 나누기
        arr = lines.split(', ')

        # 맨처음 [ 를 지우기
        arr[0] = arr[0].replace("[", "")

        # 맨 마지막 데이터가 손실되는 경우가 있음 그 경우 처리
        arr[-1] = arr[-1].replace("]\n","")
        # ,를 포함하면 없애준다
        if arr[-1].__contains__(','):
            arr[-1] = arr[-1].replace(',','')

        # 띄어쓰기 위치를 찾아서 그 전까지 나타내느 거로로
        if arr[-1].__contains__(' '):
            idx = arr[-1].find(' ')
            arr[-1] = arr[-1][:idx]

        # 맨 마지막 데이터가 짤렸을때 (기본 22 이상 되어야 정상 측정임)
        if len(arr[-1]) < 22:
            arr.pop(-1)


    # 데이터 프레임에 데이터 추가 // 시간대에 맞게 각각의 값을 넣어줍니다
    # 각각의 데이터를 태그를 보고 식별한 후 데이터가 알맞게 들어오면 각각의 데이터 프레임에 넣어줍니다.
    for l in arr:
        temp = l.split('+')

        # 레벨 부분이 숫자가 아니면
        if not is_number(temp[0]):
            continue
        # 측정된 값 부분이 숫자가 아니면
        if not is_number(temp[3]):
            continue

        # to Time from UNIX Time stamp
        if len(temp[2]) > 10:
            temp[2] = int(temp[2])/1000
        temp[2] = datetime.fromtimestamp(temp[2])

        # temp = ['1', 'AX', '1643912264855', '-3.6631858']
        # Type = AX, AY, AZ, GX, GY, GZ, HR ,SC
        if temp[1] == 'HR':
            heartR.loc[len(heartR)] = [int(temp[0]), temp[2], float(temp[3])]
        elif temp[1] == 'AX':
            accelX.loc[len(accelX)] = [int(temp[0]), temp[2], float(temp[3])]
        elif temp[1] == 'AY':
            accelY.loc[len(accelY)] = [int(temp[0]), temp[2], float(temp[3])]
        elif temp[1] == 'AZ':
            accelZ.loc[len(accelZ)] = [int(temp[0]), temp[2], float(temp[3])]
        elif temp[1] == 'GX':
            gyroX.loc[len(gyroX)] = [int(temp[0]), temp[2], float(temp[3])]
        elif temp[1] == 'GY':
            gyroY.loc[len(gyroY)] = [int(temp[0]), temp[2], float(temp[3])]
        elif temp[1] == 'GZ':
            gyroZ.loc[len(gyroZ)] = [int(temp[0]), temp[2], float(temp[3])]
        elif temp[1] == 'SC':
            stepC.loc[len(stepC)] = [int(temp[0]), temp[2], float(temp[3])]

    df_list = [accelX, accelY, accelZ, gyroX, gyroY, gyroZ]

    # 시간 중복 데이터 뒤에 부분 제거
    for i in df_list:
        i.drop_duplicates(['time'], inplace=True, ignore_index=True)

    print(accelX.shape, accelY.shape, accelZ.shape, gyroX.shape, gyroY.shape, gyroZ.shape)

    # 스탠다드 정규화 해줍니다.
    std_scaler = StandardScaler()
    for i in df_list:
        arr = np.array(i.iloc[:, 2]).reshape(-1, 1) # 2번
        arr = std_scaler.fit_transform(arr)
        i.iloc[:, 2] = arr

    ## 각 데이터프레임들을 레벨별로 합쳐 주면서 ~ 각 폴더에 저장까지 해주면서~
    # for문을 이용하여 plot 하기 위해 array로 묶어줍니다.
    data_type = [accelY, accelZ, gyroX, gyroY, gyroZ]
    total_data = accelX
    for j in data_type: # 돌아가면서 merge 수행
        total_data = pd.merge(total_data, j, on=['level', 'time'], how='inner') # level과 time을 기준으로 inner join!

    # 정제된 데이터를 레벨별로 각 폴더에 저장!!
    total_data.to_csv(current_path+"/time/"+filename+"_"+".csv", header=False, index=False)
    print(total_data)
    complete_file_len += 1




