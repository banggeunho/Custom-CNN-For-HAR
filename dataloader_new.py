import os
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import glob

# setting
data_seg = 200  # 데이터를 몇개씩 가져올지 # Sampling rate = 50Hz
window_size = 50
batch_size = 8  # 배치-사이즈

class CustomDataset(Dataset):
    def __init__(self, root_path, data_seg, mode):
        self.data = []
        self.mode = mode
        # 파일 디렉토리 가져오기
        if self.mode == 'train':
            train_path = glob.glob(root_path + '/train/**', recursive=True)
            train_paths = []
            for i in train_path:
                if os.path.isfile(i):
                    train_paths.append(i)

            for file in train_paths:
                try:
                    df = pd.read_csv(file)
                    # 지정해준 segement값 만큼 데이터를 가져와서, 라벨링 해줍니다.
                    for i in range(0, len(df.index), window_size):
                        if i + data_seg > len(df.index):  # 지정해준 값을 현재 인덱스에 더했을때 csv의 길이보다 초과하면 break
                            break
                        temp = df[i:i + data_seg]
                        temp_acc = temp.iloc[:, 0:3].transpose()
                        temp_acc = np.expand_dims(temp_acc, axis=0)
                        temp_gyro = temp.iloc[:, 3:].transpose()
                        temp_gyro = np.expand_dims(temp_gyro, axis=0)
                        temp_x = np.concatenate([temp_acc, temp_gyro], axis=0)
                        # print(temp_x.shape)
                        # print(temp_x)
                        # 파일이름에서 level(label)을 가져옵니다.
                        if file[-6] == '_':
                            temp_y = file[-5:-4]
                        else:
                            temp_y = file[-6:-4]
                        self.data.append((temp_x, int(temp_y) - 1))  # self.data list에 추가합니다.
                        # print((temp_x, int(temp_y)))
                except ValueError:
                    continue

        elif self.mode == 'val':
            val_path = glob.glob(root_path + '/val/**', recursive=True)
            val_paths = []
            for i in val_path:
                if os.path.isfile(i):
                    val_paths.append(i)

            for file in val_paths:
                try:
                    df = pd.read_csv(file)
                    # 지정해준 segement값 만큼 데이터를 가져와서, 라벨링 해줍니다.
                    for i in range(0, len(df.index), data_seg):
                        if i + data_seg > len(df.index):  # 지정해준 값을 현재 인덱스에 더했을때 csv의 길이보다 초과하면 break
                            break
                        temp = df[i:i + data_seg]
                        temp_acc = temp.iloc[:, 0:3].transpose()
                        temp_acc = np.expand_dims(temp_acc, axis=0)
                        temp_gyro = temp.iloc[:, 3:].transpose()
                        temp_gyro = np.expand_dims(temp_gyro, axis=0)
                        temp_x = np.concatenate([temp_acc, temp_gyro], axis=0)

                        # 파일이름에서 level(label)을 가져옵니다.
                        if file[-6] == '_':
                            temp_y = file[-5:-4]
                        else:
                            temp_y = file[-6:-4]
                        self.data.append((temp_x, int(temp_y) - 1))  # self.data list에 추가합니다.
                        # print((temp_x, int(temp_y)))
                except ValueError:
                    continue

        elif self.mode == 'test':
            test_path = glob.glob(root_path + '/test/**', recursive=True)
            test_paths = []
            for i in test_path:
                if os.path.isfile(i):
                    test_paths.append(i)

            for file in test_paths:
                try:
                    df = pd.read_csv(file)
                    # 지정해준 segement값 만큼 데이터를 가져와서, 라벨링 해줍니다.
                    for i in range(0, len(df.index), data_seg):
                        if i + data_seg > len(df.index):  # 지정해준 값을 현재 인덱스에 더했을때 csv의 길이보다 초과하면 break
                            break
                        temp = df[i:i + data_seg]
                        temp_acc = temp.iloc[:, 0:3].transpose()
                        temp_acc = np.expand_dims(temp_acc, axis=0)
                        temp_gyro = temp.iloc[:, 3:].transpose()
                        temp_gyro = np.expand_dims(temp_gyro, axis=0)
                        temp_x = np.concatenate([temp_acc, temp_gyro], axis=0)

                        # 파일이름에서 level(label)을 가져옵니다.
                        if file[-6] == '_':
                            temp_y = file[-5:-4]
                        else:
                            temp_y = file[-6:-4]
                        self.data.append((temp_x, int(temp_y) - 1))  # self.data list에 추가합니다.
                        # print((temp_x, int(temp_y)))
                except ValueError:
                    continue
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        # x = np.expand_dims(x, axis=0) # CNN 모델에 적용시키기 위해 차원(채널)을 하나 더 늘려줍니다.
        x = torch.FloatTensor(x)

        return x, y


current_path = os.getcwd()
root_path = './dataset/'
## data split

datasets = {}
datasets['train'] = CustomDataset(root_path, data_seg, 'train')
datasets['valid'] = CustomDataset(root_path, data_seg, 'val')
datasets['test'] = CustomDataset(root_path, data_seg, 'test')

print("Data segmentation :", data_seg)
print("Number of Training set : ", len(datasets['train']))
print("Number of Validation set : ", len(datasets['valid']))
print("Number of Test set : ", len(datasets['test']))

## data loader 선언
dataloaders = {}
dataloaders['train'] = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, drop_last=True)
dataloaders['valid'] = DataLoader(datasets['valid'], batch_size=batch_size, shuffle=False, drop_last=True)
dataloaders['test'] = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, drop_last=True)