from sklearn.metrics import confusion_matrix
from dataloader import dataloaders
from model import CNN
import torch
from barbar import Bar
import numpy as np
import copy, time
import os
from datetime import datetime
# Plot the confusion matrix
from dataPlot import plot_confusion_matrix
import matplotlib.pyplot as plt
# 랜덤 시드 고정
torch.manual_seed(777)
learning_rate = 0.001
training_epochs = 100

if torch.cuda.is_available():
  device = "cuda:0"
  print('device 0 :', torch.cuda.get_device_name(0))

model = CNN()
model.to(device)
criterion = torch.nn.CrossEntropyLoss()    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
lmbda = lambda epoch: 1
exp_lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

for epoch in range(training_epochs):
    print('Epoch {}/{}'.format(epoch, training_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss, running_corrects, num_cnt = 0.0, 0, 0

        # Iterate over data.
        for inputs, labels in Bar(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            num_cnt += len(labels)

        if phase == 'train':
            exp_lr_scheduler.step()

        epoch_loss = float(running_loss / num_cnt)
        epoch_acc = float((running_corrects.double() / num_cnt).cpu() * 100)

        if phase == 'train':
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)
        else:
            valid_loss.append(epoch_loss)
            valid_acc.append(epoch_acc)
        print(' {} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'valid' and epoch_acc > best_acc:
            best_idx = epoch
            best_acc = epoch_acc
            # best_model_wts = copy.deepcopy(model.state_dict())
            # best_model_wts = copy.deepcopy(model.module.state_dict())
            # Save model & checkpoint
            # state = {
            #     'epoch': epoch,
            #     'state_dict': model.state_dict(),
            #     'optimizer': optimizer.state_dict()
            # }

            # torch.save(model, 'D:/testmodel3/' + str(transfer) + '/' + str(epoch) + '_model.pt')
            # torch.save(state, 'D:/testmodel3/' + str(transfer) + '/' + str(epoch) + '_checkpoint.pt')

            # print('==> best model saved - %d / %.1f' % (best_idx, best_acc))

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best valid Acc: %d - %.1f' % (best_idx, best_acc))

correct = 0
total = 0
running_loss = 0
y_pred = []
y_true = []

with torch.no_grad():
    for data in Bar(dataloaders['test']):
        signal, label = data
        signal = signal.to(device)
        label = label.to(device)
        outputs = model(signal)
        _, predicted = outputs.max(1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        loss = criterion(outputs, label)  # batch의 평균 loss 출력
        running_loss += loss.item() * label.size(0)

        y_pred.extend(predicted.to("cpu").numpy())  # Save Prediction
        y_true.extend(label.to("cpu").numpy())  # Save Truth

    test_loss = running_loss / total
    test_acc = correct / total

print('test done : loss/acc : %.2f / %.1f' % (test_loss, test_acc * 100))


# Build confusion matrix
classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9','10','11','12','13')
cf_matrix = confusion_matrix(y_true, y_pred)
print(cf_matrix)
plot_confusion_matrix(cf_matrix, classes, test_acc, test_loss)

## 결과 그래프 그리기
print('best model : %d - %1.f / %.1f' % (best_idx, valid_acc[best_idx], valid_loss[best_idx]))
fig, ax1 = plt.subplots()

ax1.plot(train_acc, 'b-', label='Train_acc')
ax1.plot(valid_acc, 'r-', label='Valid_acc')
plt.plot(best_idx, valid_acc[best_idx], 'ro')
plt.legend(loc='center left', bbox_to_anchor=(1.15, 0.3))
ax1.set_xlabel('epoch')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('acc', color='k')
ax1.tick_params('y', colors='k')

ax2 = ax1.twinx()
ax2.plot(train_loss, 'g-', label='Train_loss')
ax2.plot(valid_loss, 'k-', label='Valid_loss')
plt.plot(best_idx, valid_loss[best_idx], 'ro')
ax2.set_ylabel('loss', color='k')
ax2.tick_params('y', colors='k')
plt.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))
# fig.tight_layout()
plt.show()
c_t = datetime.now()
file_name = str(c_t.month)+'_'+str(c_t.day)+'_'+str(c_t.hour)+'_'+str(c_t.minute)
plt.savefig(os.getcwd() + '/Results/train_'+file_name+'.png')
plt.clf()