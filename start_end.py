import os
from datetime import datetime
import numpy as np
import pandas as pd
from glob import glob

# 폴더 경로
path = './time_data/'
path_arr = glob(path+"**", recursive=True)
os.makedirs("./start_end_time/", exist_ok = True)

file_len = len(path_arr)
tmp = np.empty((0, 4)) # 데이터 저장(파일별로 사작 시간, 종료 시간, 측정 시간)


''' 
함수
is_number : 숫자인지 판별
'''
def is_number(num):
    try:
        float(num)
        return True  # num을 float으로 변환할 수 있는 경우
    except ValueError:  # num을 float으로 변환할 수 없는 경우
        return False

'''
main 부분
'''

for filename in path_arr:
    if filename[-3:] == 'txt':
        with open(filename) as f:
            try:
                lines = f.read()
            except UnicodeDecodeError:
                continue

            filename = filename[:-4]
            print(f'[{filename}')
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


            # 시작 시간, 종료 시간
            start_time = arr[0].split('+')[2]
            end_time = arr[-1].split('+')[2]

            if len(end_time) != 13:
                end_time = arr[-2].split('+')[2]

            start_time = datetime.fromtimestamp(float(int(start_time)/1000))
            end_time = datetime.fromtimestamp(float(int(end_time)/1000))

            filename = filename[-24:]

            measured_time = str(end_time - start_time)
            start_time = str(start_time)
            end_time = str(end_time)

            # 파일별 정보 tmp 변수에 저장
            tmp = np.append(tmp, np.array([[filename, start_time, end_time, measured_time]]), axis=0)

start_end = pd.DataFrame(tmp, columns=['watch', 'start_time', 'end_time', 'measured_time'])
start_end = start_end.sort_values(by=['start_time', 'end_time'])
start_end.to_csv('./start_end_time/start_end.csv', index=False, header=True)
print(start_end)
print(start_end.shape)
print('Done!')


