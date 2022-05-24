from glob import glob
import os
import shutil
from sklearn.model_selection import train_test_split
os.makedirs('C:/Users/Administrator/Desktop/output/val/', exist_ok=True)
os.makedirs('C:/Users/Administrator/Desktop/output/test/', exist_ok=True)

print('Total Dataset :', len(glob('C:/Users/Administrator/Desktop/output/train/**', recursive=True))-14)

for level in range(1, 14):
    # print('-------------------level : {}---------------'.format(level))

    image_path = "C:/Users/Administrator/Desktop/output/train/"+str(level)+"/"  # 이미지가 있는 디렉토리, 나는 절대 경로를 넣어두는 편이다.
    img_list = glob(f'{image_path}/*')  # 이미지 파일들의 이름 들을 읽어온 후 리스트로 저장한다.

    train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=22)
    val_img_list, test_img_list = train_test_split(val_img_list, test_size=0.5, random_state=222)

    # print('dataset split', len(train_img_list), len(val_img_list), len(test_img_list))

    os.makedirs('C:/Users/Administrator/Desktop/output/val/'+str(level)+'/', exist_ok=True)
    os.makedirs('C:/Users/Administrator/Desktop/output/test/'+str(level)+'/', exist_ok=True)

    for file_name in val_img_list:
        shutil.move(file_name, 'C:/Users/Administrator/Desktop/output/val/'+str(level)+'/' + os.path.basename(file_name))

    for file_name in test_img_list:
        shutil.move(file_name, 'C:/Users/Administrator/Desktop/output/test/'+str(level)+'/' + os.path.basename(file_name))

    # print(len(os.listdir('C:/Users/Administrator/Desktop/output/train/'+str(level)+'/')))
    # print(len(os.listdir('C:/Users/Administrator/Desktop/output/val/'+str(level)+'/')))
    # print(len(os.listdir('C:/Users/Administrator/Desktop/output/test/'+str(level)+'/')))

print('Done')
print('Train Dataset :', len(glob('C:/Users/Administrator/Desktop/output/train/**', recursive=True))-14)
print('Val Dataset :', len(glob('C:/Users/Administrator/Desktop/output/val/**', recursive=True))-14)
print('Test Dataset :', len(glob('C:/Users/Administrator/Desktop/output/test/**', recursive=True))-14)
