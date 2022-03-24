# Plot the confusion matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np
import os
from datetime import datetime

def plot_confusion_matrix(cm, classes, acc, loss, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    # Add Normalization Option 'prints pretty confusion metric with normalization option '''

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + ' loss/acc : %.2f / %.2f' % (loss, acc * 100))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    c_t = datetime.now()
    file_name = str(c_t.month)+'_'+str(c_t.day)+'_'+str(c_t.hour)+'_'+str(c_t.minute)
    plt.savefig(os.getcwd() + '/Results/conf_'+file_name+'.png')