import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import signal
import matplotlib.pyplot as plt


# 폴더 경로
path = './backup/1월/'
save_path = './january_output/'
current_path = os.getcwd()
os.makedirs(save_path, exist_ok = True)

# 레벨별로 output
for i in range(1, 14):
    os.makedirs(save_path+str(i), exist_ok = True)

length = 0
file_len = len(os.listdir(path))
complete_file_len = 1

# coding=utf-8
def is_number(num):
    try:
        float(num)
        return True  # num을 float으로 변환할 수 있는 경우
    except ValueError:  # num을 float으로 변환할 수 없는 경우
        return False

def scaling(scaler, df_list):
    for i in df_list:
        arr = np.array(i.iloc[:, 1]).reshape(-1, 1)
        arr = scaler.fit_transform(arr)
        i.iloc[:, 1] = arr

for filename in os.listdir(path):
    with open(path+filename) as f:
        try:
            lines = f.read()
        except UnicodeDecodeError:
            continue

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

    if len(arr) < 100:
        continue

    # print(arr)
    # 비어있는 데이터프레임을 만들어준다.
    HR, AX, AY, AZ, GX, GY, GZ, SC = np.empty((0,2), int), np.empty((0,2), int), np.empty((0,2), int), np.empty((0,2), int), np.empty((0,2), int), np.empty((0,2), int), np.empty((0,2), int), np.empty((0,2), int)

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
        # print(temp)
        # temp = ['1', 'AX', '1643912264855', '-3.6631858']
        # Type = AX, AY, AZ, GX, GY, GZ, HR ,SC
        # temp[2] = datetime.fromtimestamp(float(temp[2])/1000)
        # print(temp)
        if temp[1] == 'heartR':
            HR = np.append(HR, np.array([[int(temp[0]), float(temp[3])]]), axis = 0)
        elif temp[1] == 'accelX':
            AX = np.append(AX, np.array([[int(temp[0]), float(temp[3])]]), axis = 0)
        elif temp[1] == 'accelY':
            AY = np.append(AY, np.array([[int(temp[0]), float(temp[3])]]), axis = 0)
        elif temp[1] == 'accelZ':
            AZ = np.append(AZ, np.array([[int(temp[0]), float(temp[3])]]), axis = 0)
        elif temp[1] == 'gyroX':
            GX = np.append(GX, np.array([[int(temp[0]), float(temp[3])]]), axis = 0)
        elif temp[1] == 'gyroY':
            GY = np.append(GY, np.array([[int(temp[0]), float(temp[3])]]), axis = 0)
        elif temp[1] == 'gyroZ':
            GZ = np.append(GZ, np.array([[int(temp[0]), float(temp[3])]]), axis = 0)
        elif temp[1] == 'stepC':
            SC = np.append(SC, np.array([[int(temp[0]), float(temp[3])]]), axis = 0)

    heartR = pd.DataFrame(HR, columns=['level', 'value'])
    accelX = pd.DataFrame(AX, columns=['level', 'value'])
    accelY = pd.DataFrame(AY, columns=['level', 'value'])
    accelZ = pd.DataFrame(AZ, columns=['level', 'value'])
    gyroX = pd.DataFrame(GX, columns=['level', 'value'])
    gyroY = pd.DataFrame(GY, columns=['level', 'value'])
    gyroZ = pd.DataFrame(GZ, columns=['level', 'value'])
    stepC = pd.DataFrame(SC, columns=['level', 'value'])

    # print(heartR.shape, accelX.shape, accelY.shape, accelZ.shape, gyroX.shape, gyroY.shape, gyroZ.shape, stepC.shape)

    # 레벨별 길이 구하기
    level_num = [i for i in range(1, 14)]
    level_len = []
    temp = 0

    # pd.set_option('display.max_row', 500)

    df_list = [accelX, accelY, accelZ, gyroX, gyroY, gyroZ]
    level_df = [pd.DataFrame()] * 14
    re_level_df = [pd.DataFrame()] * 14
    original_df = pd.DataFrame()
    final_df = pd.DataFrame()
    column_names = ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']

    # Copy the original data and data removed outlier to new dataframe for scaling
    std_accelX, std_accelY, std_accelZ, std_gyroX, std_gyroY, std_gyroZ = accelX.copy(), accelY.copy(), accelZ.copy(), gyroX.copy(), gyroY.copy(), gyroZ.copy()
    std_list = [std_accelX, std_accelY, std_accelZ, std_gyroX, std_gyroY, std_gyroZ]

    # Standard Scaling
    scaling(StandardScaler(), std_list)

    # 레벨별로 데이터 나누기
    for i in level_num:
        for j in std_list:
            temp_data = j[j['level'] == i]['value'].copy()
            temp_data.reset_index(drop=True, inplace=True)
            level_df[i] = pd.concat([level_df[i], temp_data], axis=1)

    # 레벨별로 나뉜 데이터 리샘플링(down)하기
    for i in level_num:
        if len(level_df[i]) <= 0:
            continue

        min_column_cnt = int(1e9)
        level_df[i].columns = column_names
        for j in column_names:
            if level_df[i][j].count() > 0:
                min_column_cnt = min(level_df[i][j].count(), min_column_cnt)

        # print(i,'//',min_column_cnt)

        for j in column_names:
            arr = np.array(level_df[i][j]).reshape(-1, 1)
            arr = arr[np.logical_not(np.isnan(arr))]
            f = signal.resample(arr, min_column_cnt)
            re_level_df[i] = pd.concat([re_level_df[i], pd.DataFrame(f)], axis=1)

        re_level_df[i].columns = column_names
        re_level_df[i]['level'] = [i for _ in range(len(re_level_df[i]))]

        original_df = pd.concat([original_df, level_df[i]], axis=0)
        final_df = pd.concat([final_df, re_level_df[i]], axis=0)  # 최종 리샘플링된 데이터

    original_df.reset_index(drop=True, inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    # print(final_df)
    # print(accelX)
    print("기존 데이터 길이, 리샘플링한 데이터 길이 비교")
    print(accelX.shape, final_df.shape)

    # # 리샘플링 잘 되었는지 확인
    # for idx, (i, j) in enumerate(zip(column_names, std_list)):
    #     plt.plot(j.index, j['value'], 'g-', final_df[i].index, final_df[i], 'b-')
    #     plt.legend(['data', 'resampled'], loc='best')
    #     plt.show()
    # plt.clf()

    # 레벨별로 나누어서 csv 파일로 데이터 저장
    for i in level_num:  # 레벨별로 확인해서 쪼개겠습니다.
        save_data = final_df[final_df['level'] == i].copy()  # 첫 기준 데이터를 accelX로 지정해준다.
        if len(save_data) <= 0:
            continue
        save_data.reset_index(drop=True, inplace=True)
        save_data.drop(['level'], axis=1, inplace=True)  # 레벨 column 삭제
        # 정제된 데이터를 레벨별로 각 폴더에 저장!!
        save_data.to_csv(save_path + str(i) + "/" + filename + "_" + str(i) + ".csv", header=False, index=False)
        length += len(save_data)

    complete_file_len += 1

print('Total length of data : ', length)
print('Done!')



