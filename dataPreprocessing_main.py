from dataPreprocessing_new import february_preprocessing
from dataPreprocessing_1 import january_preprocessing
import os
from train_test_split import train_val_test_split

low_cutoffs = [2.5, 5., 7.5]
high_cutoffs = [10., 15., 20., 24.5]
total_length = 0

for low_cutoff in low_cutoffs:
    for high_cutoff in high_cutoffs:
        save_path = f'./filtered_output_ss/{low_cutoff}-{high_cutoff}/'

        print(f'low : {low_cutoff} high : {high_cutoff}')
        print(save_path)

        os.makedirs(save_path, exist_ok=True)

        # 레벨별로 output
        for i in range(1, 14):
            os.makedirs(save_path + str(i), exist_ok=True)

        total_length += january_preprocessing(save_path, low_cutoff, high_cutoff)
        total_length += february_preprocessing(save_path, low_cutoff, high_cutoff)

        train_val_test_split(save_path, 13)


print(f'Total length : {total_length}')
print('Done!')


