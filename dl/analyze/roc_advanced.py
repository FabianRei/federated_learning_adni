import os
from matplotlib import pyplot as plt
import numpy
import pickle


def get_data(fpath):
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    fpr = data['fpr']
    tpr = data['tpr']
    thres = data['thres']
    roc_auc = data['roc_auc']
    return fpr, tpr, thres, roc_auc


fp1 = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds_lower_lr_imgnet_fixed_experiment\seed_11\one_slice'
file1 = 'roc_09-23_17-50_lr_0_0001_pretrained_30epochs_rod_0_1_da_10_roc_data.p'
file1 = 'roc_09-23_17-50_lr_0_0001_pretrained_reg_30epochs_rod_0_1_da_10_roc_data.p'
full1 = os.path.join(fp1, file1)

fp2 = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds_lower_lr_imgnet_fixed_experiment\seed_11\dist_10'
file2 = 'roc_09-23_17-33_lr_0_0001_pretrained_30epochs_rod_0_1_da_10_roc_data.p'
file2 = 'roc_09-23_17-33_lr_0_0001_pretrained_reg_30epochs_rod_0_1_da_10_roc_data.p'
full2 = os.path.join(fp2, file2)

fpr1, tpr1, thres1, roc_auc1 = get_data(full1)
fpr2, tpr2, thres2, roc_auc2 = get_data(full2)

out_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds_lower_lr_imgnet_fixed_experiment'
fname = 'reg.png'
fname = os.path.join(out_path, fname)

lw = 2
plt.plot(fpr1, tpr1,
         lw=lw, label='One slice ROC curve (area = %0.3f)' % roc_auc1)
plt.plot(fpr2, tpr2,
         lw=lw, label='Three slice ROC curve (area = %0.3f)' % roc_auc2)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(which='both')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig(fname, dpi=200)
# (os.path.join(out_path, f'{fname}.png')
plt.clf()