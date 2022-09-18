from dataPreprocessing_svm_1 import january_preprocessing
from dataPreprocessing_svm_2 import february_preprocessing
import os
from train_test_split import train_val_test_split

total_length = 0


save_path = f'./진우형님/1월, 2월 인풋 데이터/'
path_1 = './backup/1월/'
path_2 = './backup/2월/'
train_rate = 0.6
val_rate = 0.2
num_classes = 13
#
# print(save_path)
# os.makedirs(save_path, exist_ok=True)
#
# # 레벨별로 output
# for i in range(1, 14):
#     os.makedirs(save_path + str(i), exist_ok=True)
#
# total_length += january_preprocessing(path_1, save_path)
# total_length += february_preprocessing(path_2, save_path)

train_val_test_split(save_path, num_classes, train_rate, val_rate)


print(f'Total length : {total_length}')
print('Done!')


