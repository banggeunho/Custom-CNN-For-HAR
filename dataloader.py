
import os
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import glob


# setting
data_seg = 50    # 데이터를 몇개씩 가져올지
batch_size = 64  # 배치-사이즈


class CustomDataset(Dataset):
    def __init__(self, data_path, data_seg):
        self.data = []

        for file in data_path:
            # print(file)
            try:
                df = pd.read_csv(file)
                # 지정해준 segement값 만큼 데이터를 가져와서, 라벨링 해줍니다.
                for i in range(0, len(df.index), data_seg):
                    if i+data_seg > len(df.index): # 지정해준 값을 현재 인덱스에 더했을때 csv의 길이보다 초과하면 break
                        break
                    temp = df[i:i+data_seg]
                    temp_acc = temp.iloc[:, 0:3].transpose()
                    temp_acc = np.expand_dims(temp_acc, axis=0)
                    temp_gyro = temp.iloc[:, 3:].transpose()
                    temp_gyro = np.expand_dims(temp_gyro, axis=0)
                    temp_x = np.concatenate([temp_acc, temp_gyro], axis=0)
                    print(temp_x.shape)
                    # print(temp_x)
                    # 파일이름에서 level(label)을 가져옵니다.
                    if file[-6] == '_':
                        temp_y = file[-5:-4]
                    else:
                        temp_y = file[-6:-4]
                    self.data.append((temp_x, int(temp_y)-1)) # self.data list에 추가합니다.
                    print((temp_x, int(temp_y)))
            except ValueError:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        # x = np.expand_dims(x, axis=0) # CNN 모델에 적용시키기 위해 차원(채널)을 하나 더 늘려줍니다.
        x = torch.FloatTensor(x)
        # y = torch.IntTensor(y)
        return x, y

current_path = os.getcwd()
train_path = glob.glob(current_path + '/output/**', recursive=True)
data_path = []
for i in train_path:
    if os.path.isfile(i):
        data_path.append(i)



train = CustomDataset(data_path, data_seg)

## data split
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
datasets={}
train_idx, temp_idx = train_test_split(list(range(len(train))), test_size=0.2, random_state=555)
datasets['train'] = Subset(train, train_idx)
temp_dataset = Subset(train, temp_idx)
valid_idx, test_idx = train_test_split(list(range(len(temp_dataset))), test_size=0.5, random_state=555)
datasets['valid'] = Subset(temp_dataset, valid_idx)
datasets['test'] = Subset(temp_dataset, test_idx)

print("Number of Training set : ", len(datasets['train']))
print("Number of  Validation set : ", len(datasets['valid']))
print("Number of Test set : ", len(datasets['test']))



## data loader 선언
dataloaders = {}
dataloaders['train'] = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, drop_last=True)
dataloaders['valid'] = DataLoader(datasets['valid'], batch_size=batch_size, shuffle=False, drop_last=True)
dataloaders['test']  = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, drop_last=True)

