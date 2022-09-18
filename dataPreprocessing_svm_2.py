import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from glob import glob
from tqdm.auto import tqdm
import math


def february_preprocessing(path, save_path):
    # 폴더 경로

    length = 0
    complete_file_len = 1

    ''' 함수 구현
    is_number : 숫자인지 판별
    scaling : standard scaling
    plot6values : 데이터 플롯
    '''
    def is_number(num):
        try:
            float(num)
            return True  # num을 float으로 변환할 수 있는 경우
        except ValueError:  # num을 float으로 변환할 수 없는 경우
            return False

    def scaling(df, scaler, columns):
        for column in columns:
            arr = np.array(df[column]).reshape(-1, 1)
            arr = scaler.fit_transform(arr)
            df[column] = arr

        return df

    def SVM_algorithm(x, y, z):
        result = math.sqrt(math.pow(x, 2) + math.pow(y, 2) + math.pow(z, 2))
        return result


    from scipy.signal import butter, filtfilt

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y


    '''
    main 부분
    레벨별로 폴더에 접근해서 입력데이터로 바꿔주기
    '''
    for filename in tqdm(os.listdir(path)):
        with open(path+filename) as f:
            try:
                lines = f.read()
            except UnicodeDecodeError:
                continue

            filename = filename[:-4]
            # print(f'[{complete_file_len}/{file_len}, {complete_file_len/file_len * 100:.2f}%]{filename}')
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

        # 비어있는 데이터프레임을 만들어준다.
        HR, AX, AY, AZ, GX, GY, GZ, SC = np.empty((0,2), int), np.empty((0,2), int), np.empty((0,2), int),\
                                         np.empty((0,2), int), np.empty((0,2), int), np.empty((0,2), int),\
                                         np.empty((0,2), int), np.empty((0,2), int)

        # 데이터 프레임에 데이터 추가 // 시간대에 맞게 각각의 값을 넣어줍니다
        # 각각의 데이터를 태그를 보고 식별한 후 데이터가 알맞게 들어오면 각각의 데이터 프레임에 넣어줍니다.
        for l in range(150, len(arr)):
            temp = arr[l].split('+')

            # 레벨 부분이 숫자가 아니면
            if not is_number(temp[0]):
                continue
            # 측정된 값 부분이 숫자가 아니면
            if not is_number(temp[3]):
                continue

            # temp = ['1', 'AX', '1643912264855', '-3.6631858']
            # Type = AX, AY, AZ, GX, GY, GZ, HR ,SC
            # temp[2] = datetime.fromtimestamp(float(temp[2])/1000)
            # print(temp)
            if temp[1] == 'HR':
                HR = np.append(HR, np.array([[float(temp[0]), float(temp[3])]]), axis = 0)
            elif temp[1] == 'AX':
                AX = np.append(AX, np.array([[float(temp[0]), float(temp[3])]]), axis = 0)
            elif temp[1] == 'AY':
                AY = np.append(AY, np.array([[float(temp[0]), float(temp[3])]]), axis = 0)
            elif temp[1] == 'AZ':
                AZ = np.append(AZ, np.array([[float(temp[0]), float(temp[3])]]), axis = 0)
            elif temp[1] == 'GX':
                GX = np.append(GX, np.array([[float(temp[0]), float(temp[3])]]), axis = 0)
            elif temp[1] == 'GY':
                GY = np.append(GY, np.array([[float(temp[0]),float(temp[3])]]), axis = 0)
            elif temp[1] == 'GZ':
                GZ = np.append(GZ, np.array([[float(temp[0]),float(temp[3])]]), axis = 0)
            elif temp[1] == 'SC':
                SC = np.append(SC, np.array([[float(temp[0]), float(temp[3])]]), axis = 0)

        # print(HR.shape, AX.shape, AY.shape, AZ.shape, GX.shape, GY.shape, GZ.shape, SC.shape)

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
        column_names = ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ', 'F_AX', 'F_AY', 'F_AZ', 'F_GX', 'F_GY', 'F_GZ', 'A_SVM', 'G_SVM']


        filtered_accelX = pd.DataFrame(columns=['level', 'value'])
        filtered_accelY = pd.DataFrame(columns=['level', 'value'])
        filtered_accelZ = pd.DataFrame(columns=['level', 'value'])
        filtered_gyroX = pd.DataFrame(columns=['level', 'value'])
        filtered_gyroY = pd.DataFrame(columns=['level', 'value'])
        filtered_gyroZ = pd.DataFrame(columns=['level', 'value'])

        filtered_df = [filtered_accelX, filtered_accelY, filtered_accelZ, filtered_gyroX, filtered_gyroY, filtered_gyroZ]
        filtered_columns = ['F_AX', 'F_AY', 'F_AZ', 'F_GX', 'F_GY', 'F_GZ']

        for idx, data in enumerate(df_list):
            # setting
            order = 6
            fs = 50.
            cutoff = 3.
            low_cutoff = 2.5
            high_cutoff = 24.5

            # fig = plt.figure()
            # fig.tight_layout()
            # fig.suptitle(column_names[idx])
            # k = 1

            for i in level_num:
                temp = data[data['level'] == i].copy()[150:-150]
                filtered_temp = temp.copy()
                if len(temp) <= 50:
                    continue

                bpf = butter_bandpass_filter(temp['value'], low_cutoff, high_cutoff, fs, order)
                filtered_temp['value'] = bpf
                t = np.arange(0, len(temp))
                filtered_df[idx] = pd.concat([filtered_df[idx], filtered_temp], axis=0)

                # ##### plot 하는부분  #######
                # if i < 5:
                #     plt.title
                #     ax = fig.add_subplot(2, 2, k)
                #     ax.plot(t, temp['value'])
                #     ax.plot(t, bpf, 'r')
                #     ax.set_title(f'level = {i}')
                #     k+=1

            # plt.tight_layout()
            # plt.show()
            filtered_df[idx].reset_index(drop = True, inplace = True)


        def Resampling(data_list, columns, mode='original'):
            level_df = [pd.DataFrame()]*14
            re_level_df = [pd.DataFrame()]*14
            original_df = pd.DataFrame()
            final_df = pd.DataFrame()

            # 레벨별로 데이터 나누기
            for i in level_num:
                for j in data_list:
                    if mode == 'original':
                        temp_data = j[j['level'] == i]['value'].copy()[150:-150]
                    else:
                        temp_data = j[j['level'] == i]['value'].copy()
                    temp_data.reset_index(drop=True, inplace=True)
                    level_df[i] = pd.concat([level_df[i], temp_data], axis=1)

            # 레벨별로 나뉜 데이터 리샘플링(down)하기
            for i in level_num:
                if len(level_df[i]) <= 0:
                    continue

                min_column_cnt = int(1e9)
                level_df[i].columns = columns
                for j in columns:
                    if level_df[i][j].count() > 0:
                        min_column_cnt = min(level_df[i][j].count(), min_column_cnt)

                # print(i,'//',min_column_cnt)

                for j in columns:
                    arr = np.array(level_df[i][j]).reshape(-1, 1)

                    arr = arr[np.logical_not(pd.isnull(arr))]
                    # print(arr)
                    f = signal.resample(arr, min_column_cnt)
                    re_level_df[i] = pd.concat([re_level_df[i], pd.DataFrame(f)], axis = 1)

                re_level_df[i].columns = columns
                if mode == 'filtered':
                    re_level_df[i]['level'] = [i for _ in range(len(re_level_df[i]))]

                original_df = pd.concat([original_df, level_df[i]], axis=0)
                final_df = pd.concat([final_df, re_level_df[i]], axis=0) # 최종 리샘플링된 데이터

            original_df.reset_index(drop=True, inplace=True)
            final_df.reset_index(drop=True, inplace=True)
            return final_df

        # print(final_df)
        # print(accelX)
        # print("기존 데이터 길이, 리샘플링한 데이터 길이 비교")
        # print(accelX.shape, final_df.shape)

        columns = ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']
        filtered_columns = ['F_AX', 'F_AY', 'F_AZ', 'F_GX', 'F_GY', 'F_GZ']
        result_df = pd.concat([Resampling(df_list, columns, 'original'), Resampling(filtered_df, filtered_columns, 'filtered')], axis = 1)

        # print(result_df)
        # print(result_df.shape)
        # print(result_df.columns)

        accel_svm = np.empty((0, 1), float)
        gyro_svm = np.empty((0, 1), float)
        f_accel_svm = np.empty((0, 1), float)
        f_gyro_svm = np.empty((0, 1), float)

        for i in range(len(result_df)):
            accel_svm = np.append(accel_svm, SVM_algorithm(result_df['AX'][i].item(), result_df['AY'][i].item(), result_df['AZ'][i].item()))
            gyro_svm = np.append(gyro_svm, SVM_algorithm(result_df['GX'][i].item(), result_df['GY'][i].item(), result_df['GZ'][i].item()))
            f_accel_svm = np.append(f_accel_svm, SVM_algorithm(result_df['F_AX'][i].item(), result_df['F_AY'][i].item(), result_df['F_AZ'][i].item()))
            f_gyro_svm = np.append(f_gyro_svm, SVM_algorithm(result_df['F_GX'][i].item(), result_df['F_GY'][i].item(), result_df['F_GZ'][i].item()))

        result_df['A_SVM'] = accel_svm
        result_df['G_SVM'] = gyro_svm
        result_df['F_A_SVM'] = f_accel_svm
        result_df['F_G_SVM'] = f_gyro_svm

        # Standard Scaling
        column_names = ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ', 'F_AX', 'F_AY', 'F_AZ', 'F_GX', 'F_GY', 'F_GZ', 'A_SVM', 'G_SVM']
        # column_names = ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ', 'F_AX', 'F_AY', 'F_AZ', 'F_GX', 'F_GY', 'F_GZ', 'A_SVM', 'G_SVM']
        final_df = scaling(result_df, StandardScaler(), column_names)

        # # 리샘플링 잘 되었는지 확인
        # for idx, (i, j) in enumerate(zip(column_names, df_list)):
        #     plt.plot(j.index, j['value'], 'g-', final_df[i].index, final_df[i], 'b-')
        #     plt.legend(['data', 'resampled'], loc='best')
        #     plt.show()
        # plt.clf()

        # 레벨별로 나누어서 csv 파일로 데이터 저장
        for i in level_num: # 레벨별로 확인해서 쪼개겠습니다.
            save_data = final_df[final_df['level'] == i].copy() #첫 기준 데이터를 accelX로 지정해준다.
            if len(save_data) == 0:
                continue
            save_data.reset_index(drop=True, inplace=True)
            save_data.drop(['level'], axis=1, inplace=True) # 레벨 column 삭제
            # 정제된 데이터를 레벨별로 각 폴더에 저장!!
            save_data.to_csv(save_path+str(i)+"/"+filename+"_"+str(i)+".csv", index=False)
            length += len(save_data)

        complete_file_len += 1

    return length