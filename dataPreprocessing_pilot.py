import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import signal
import matplotlib.pyplot as plt
from glob import glob

# 폴더 경로
src_path = './pilot_data/'
save_path = './pilot_output/'
current_path = os.getcwd()
os.makedirs(save_path, exist_ok = True)

# 레벨별로 output
for i in range(1, 11):
    os.makedirs(save_path+str(i), exist_ok = True)

length = 0
file_len = len(glob(src_path+'**', recursive=True))-11
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


'''
main 부분
레벨별로 폴더에 접근해서 입력데이터로 바꿔주기
'''
for idx in range(1, 11):
    for filename in os.listdir(src_path+str(idx)+'/'):

        with open(src_path+str(idx)+'/'+filename) as f:
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

        # 비어있는 데이터프레임을 만들어준다.
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
            elif temp[0] == 'accelX':
                AX = np.append(AX, np.array([[idx, float(temp[2])]]), axis = 0)
            elif temp[0] == 'accelY':
                AY = np.append(AY, np.array([[idx, float(temp[2])]]), axis = 0)
            elif temp[0] == 'accelZ':
                AZ = np.append(AZ, np.array([[idx, float(temp[2])]]), axis = 0)
            elif temp[0] == 'gyroX':
                GX = np.append(GX, np.array([[idx, float(temp[2])]]), axis = 0)
            elif temp[0] == 'gyroY':
                GY = np.append(GY, np.array([[idx, float(temp[2])]]), axis = 0)
            elif temp[0] == 'gyroZ':
                GZ = np.append(GZ, np.array([[idx, float(temp[2])]]), axis = 0)
            elif temp[0] == 'stepC':
                SC = np.append(SC, np.array([[idx, float(temp[2])]]), axis = 0)

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

        filtered_df = pd.DataFrame()

        # fig = plt.figure()
        # fig.tight_layout()
        # k = 1

        for column, data in zip(column_names, df_list):
            # setting
            order = 6
            fs = 50.
            cutoff = 3.
            low_cutoff = 2.5
            high_cutoff = 24.5

            temp = data['value'].copy()[100:-100]
            if len(temp) <= 50:
                continue
            # temp.reset_index(drop=True, inplace=True)

            # lpf = PassFilter(cutoff_freq=0.1, ts=0.02, init_value=temp['value'][0])
            # filtered_data = [lpf.filter(data) for data in temp['value']]
            # lpf2 = butter_lowpass_filter(temp['value'], cutoff, fs, order)

            bpf = butter_bandpass_filter(temp, low_cutoff, high_cutoff, fs, order)
            kkk = pd.DataFrame(bpf, columns = [column])
            t = np.arange(0, len(temp))
            filtered_df = pd.concat([filtered_df, kkk], axis=1)

        #     ##### plot 하는부분  #######
        #     plt.title
        #     ax = fig.add_subplot(3, 2, k)
        #     ax.plot(t, temp)
        #     ax.plot(t, bpf, 'r')
        #
        #     k+=1
        #
        #
        # plt.tight_layout()
        # plt.show()


        filtered_df.reset_index(drop=True, inplace=True)
        # print(filtered_df)

        if len(filtered_df) > 0:
            min_column_cnt = int(1e9)
            for i in column_names:
                if filtered_df[i].count() > 0:
                    min_column_cnt = min(filtered_df[i].count(), min_column_cnt)
            # print(i,'//',min_column_cnt)

            # 데이터 resampling한 후 final_df에 저장
            for i in column_names:
                arr = np.array(filtered_df[i]).reshape(-1, 1)
                arr = arr[np.logical_not(np.isnan(arr))]
                f = signal.resample(arr, min_column_cnt)
                final_df = pd.concat([final_df, pd.DataFrame(f)], axis = 1)

            final_df.columns = column_names
            final_df.reset_index(drop=True, inplace=True)

            # print(final_df)

            # Standard Scaling
            final_df = scaling(final_df, StandardScaler(), column_names)
            #
            print(final_df.shape)

            # # plot6Valeus(final_df, level_len, 2)
            # # plt.show()
            #
            # #리샘플링 잘 되었는지 확인
            # for (i, j) in zip(column_names, df_list):
            #     plt.plot(j.index, j['value'], 'g-', final_df[i].index, final_df[i], 'b-')
            #     plt.legend(['data', 'resampled'], loc='best')
            #     plt.show()
            # plt.clf()

            # final_df의 데이터를 레벨별로 각 폴더에 저장!!
            final_df.to_csv(save_path+str(idx)+"/"+filename+"_"+str(idx)+".csv", header=False, index=False)
            length += len(final_df)
            complete_file_len += 1

print('Total length of data : ', length)

from train_test_split import train_val_test_split
train_val_test_split(save_path, 10)
print('Done!')


