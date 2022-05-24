import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import signal
import matplotlib.pyplot as plt
from glob import glob

# 경로 안에 있는 모든 파일 읽어온다.
# 여기 경로 수정해서 사용하세용~
path = './pilot_data/'
current_path = os.getcwd()
os.makedirs("./pilot_output/", exist_ok = True)
# 폴더를 만들어줍니다.
for i in range(1, 11):
    os.makedirs(current_path + "/pilot_output/"+str(i), exist_ok = True)

length = 0
file_len = len(glob('./pilot_data/**', recursive=True))-11)
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

def plot6Valeus(df, seq, column_names):
    data_type_name = ['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ']
    plt.figure(seq, figsize=(15, 8))  # 먼저 창을 만들고
    n = 1
    # plt.suptitle(filename + '   level: ' + level, fontsize=15)
    for i, data in enumerate(column_names):
        ax = plt.subplot(2, 3, n)  # for문을 돌면서 Axes를 추가
        plt.title("%s " % data_type_name[i], fontsize=15)
        ax.plot(df.index, df[data], "-", label=str(data), alpha=.6)  # 그래프 추가
        plt.tick_params(width=0)
        n += 1
    plt.tight_layout()  # 창 크기에 맞게 조정

for idx in range(1, 11):
    for filename in os.listdir(path+'/'+str(idx)+'/'):

        with open(path+'/'+str(idx)+'/'+filename) as f:
            try:
                lines = f.read()
            except UnicodeDecodeError:
                continue

            level = idx
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

        if len(arr) < 300:
            continue

        HR, AX, AY, AZ, GX, GY, GZ, SC = np.empty((0,2), int), np.empty((0,2), int), np.empty((0,2), int), np.empty((0,2), int), np.empty((0,2), int), np.empty((0,2), int), np.empty((0,2), int), np.empty((0,2), int)
        # 데이터 프레임에 데이터 추가 // 시간대에 맞게 각각의 값을 넣어줍니다
        # 각각의 데이터를 태그를 보고 식별한 후 데이터가 알맞게 들어오면 각각의 데이터 프레임에 넣어줍니다.
        for l in arr:
            temp = l.split('+')

            # 레벨 부분이 숫자가 아니면
            # if not is_number(temp[0]):
            #     continue
            # 측정된 값 부분이 숫자가 아니면
            if not is_number(temp[2]):
                continue

            # temp = ['1', 'AX', '1643912264855', '-3.6631858']
            # Type = AX, AY, AZ, GX, GY, GZ, HR ,SC
            # temp[2] = datetime.fromtimestamp(float(temp[2])/1000)
            # print(temp)

            if temp[0] == 'heartR':
                HR = np.append(HR, np.array([[idx, float(temp[2])]]), axis = 0)
                # heartR.loc[len(heartR)] = [idx, float(temp[2])]
            elif temp[0] == 'accelX':
                AX = np.append(AX, np.array([[idx, float(temp[2])]]), axis = 0)
                # accelX.loc[len(accelX)] = [idx, float(temp[2])]
            elif temp[0] == 'accelY':
                AY = np.append(AY, np.array([[idx, float(temp[2])]]), axis = 0)
                # accelY.loc[len(accelY)] = [idx, float(temp[2])]
            elif temp[0] == 'accelZ':
                AZ = np.append(AZ, np.array([[idx, float(temp[2])]]), axis = 0)
                # accelZ.loc[len(accelZ)] = [idx, float(temp[2])]
            elif temp[0] == 'gyroX':
                GX = np.append(GX, np.array([[idx, float(temp[2])]]), axis = 0)
                # gyroX.loc[len(gyroX)] = [idx, float(temp[2])]
            elif temp[0] == 'gyroY':
                GY = np.append(GY, np.array([[idx, float(temp[2])]]), axis = 0)
                # gyroY.loc[len(gyroY)] = [idx, float(temp[2])]
            elif temp[0] == 'gyroZ':
                GZ = np.append(GZ, np.array([[idx, float(temp[2])]]), axis = 0)
                # gyroZ.loc[len(gyroZ)] = [idx, float(temp[2])]
            elif temp[0] == 'stepC':
                SC = np.append(SC, np.array([[idx, float(temp[2])]]), axis = 0)
                # stepC.loc[len(stepC)] = [idx, float(temp[2])]

        # print(heartR,"\n", accelX, "\n", accelY, "\n", accelZ, "\n", gyroX, "\n", gyroY, "\n", gyroZ, "\n", stepC)
        # pd.set_option('display.max_row', 500)

        print(HR.shape, AX.shape, AY.shape, AZ.shape, GX.shape, GY.shape, GZ.shape, SC.shape)

        heartR = pd.DataFrame(HR, columns=['level', 'value'])
        accelX = pd.DataFrame(AX, columns=['level', 'value'])
        accelY = pd.DataFrame(AY, columns=['level', 'value'])
        accelZ = pd.DataFrame(AZ, columns=['level', 'value'])
        gyroX = pd.DataFrame(GX, columns=['level', 'value'])
        gyroY = pd.DataFrame(GY, columns=['level', 'value'])
        gyroZ = pd.DataFrame(GZ, columns=['level', 'value'])
        stepC = pd.DataFrame(SC, columns=['level', 'value'])

        # print(heartR.shape, accelX.shape, accelY.shape, accelZ.shape, gyroX.shape, gyroY.shape, gyroZ.shape, stepC.shape)

        df_list = [accelX, accelY, accelZ, gyroX, gyroY, gyroZ]
        original_df = pd.DataFrame()
        concat_df = pd.DataFrame()
        re_df = pd.DataFrame()
        final_df = pd.DataFrame()

        column_names = ['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ']

        # Copy the original data and data removed outlier to new dataframe for scaling
        std_accelX, std_accelY, std_accelZ, std_gyroX, std_gyroY, std_gyroZ = accelX.copy(), accelY.copy(), accelZ.copy(), gyroX.copy(), gyroY.copy(), gyroZ.copy()
        std_list = [std_accelX, std_accelY, std_accelZ, std_gyroX, std_gyroY, std_gyroZ]

        # Standard Scaling
        scaling(StandardScaler(), std_list)

        # 레벨별로 데이터 나누기

        for j in std_list:
            temp_data = j['value'].copy()
            temp_data.reset_index(drop=True, inplace=True)
            concat_df = pd.concat([concat_df, temp_data], axis=1)


        min_column_cnt = int(1e9)
        concat_df.columns = column_names
        for i in column_names:
            if concat_df[i].count() > 0:
                min_column_cnt = min(concat_df[i].count(), min_column_cnt)

        # print(i,'//',min_column_cnt)

        for i in column_names:
            arr = np.array(concat_df[i]).reshape(-1, 1)
            arr = arr[np.logical_not(np.isnan(arr))]
            f = signal.resample(arr, min_column_cnt)
            final_df = pd.concat([final_df, pd.DataFrame(f)], axis = 1)

        final_df.columns = column_names
        final_df.reset_index(drop=True, inplace=True)

        print(final_df.shape)


        # # plot6Valeus(final_df, level_len, 2)
        # # plt.show()
        #
        # 리샘플링 잘 되었는지 확인
        # for idx, (i, j) in enumerate(zip(column_names, std_list)):
        #     plt.plot(j.index, j['value'], 'g-', final_df[i].index, final_df[i], 'b-')
        #     plt.legend(['data', 'resampled'], loc='best')
        #     plt.show()
        # plt.clf()

        # 정제된 데이터를 레벨별로 각 폴더에 저장!!
        final_df.to_csv("./pilot_output/"+str(idx)+"/"+filename+"_"+str(idx)+".csv", header=False, index=False)
        length += len(final_df)
        complete_file_len += 1

print('Total length of data : ', length)
print('Done!')


