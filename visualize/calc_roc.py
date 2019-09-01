import pickle
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import os
from glob import glob
import re
from dl.data.logging import CsvWriter



def calc_rocs_epoch(folder, epoch=29):
    f_paths = glob(f'{folder}\*epoch_{epoch}_pred_labels*.p')
    f_paths = [p for p in f_paths if re.match(r'.*reg.*|.*20bins.*', p) is None]
    for fp in f_paths:
        with open(fp, 'rb') as f:
            tt = pickle.load(f)
            tt = tt['test']
        fname = fp[:-2] + "_roc.png"
        calc_roc(tt, fname)


def calc_roc(tt, fname):
    pred = tt[:,3]
    lab = tt[:,1]
    fpr, tpr, thres = roc_curve(lab, pred)
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
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


def create_number_csv(folder, epoch=29):
    f_paths = glob(f'{folder}\*epoch_{epoch}_pred_labels*.p')
    f_paths = [p for p in f_paths if re.match(r'.*reg.*|.*20bins.*', p) is None]
    writer = CsvWriter(os.path.join(folder, 'key_values.csv'), header=['name', 'sensitivity', 'specificity', 'accuracy'])
    for fp in f_paths:
        with open(fp, 'rb') as f:
            tt = pickle.load(f)
            tt = tt['test']
        fname = os.path.basename(fp)[:-2]
        spezi = calc_specificity(tt)
        sensi = calc_sensitivity(tt)
        acc = np.mean(tt[:,0]==tt[:,1])
        writer.write_row(name = fname, sensitivity=sensi, specificity=spezi, accuracy=acc)


def calc_specificity(tt):
    pred = tt[:, 0]
    lab = tt[:, 1]
    specificity = np.sum((pred == 0) & (lab == 0))/(np.sum((pred == 0) & (lab == 0))+np.sum((pred == 1) & (lab == 0)))
    return specificity


def calc_sensitivity(tt):
    pred = tt[:, 0]
    lab = tt[:, 1]
    sensitivity = np.sum((pred == 1) & (lab == 1))/(np.sum((pred == 1) & (lab == 1))+np.sum((pred == 0) & (lab == 1)))
    return sensitivity

if __name__ == '__main__':
    fpath = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments'
    paths = glob(fpath + r"\*site*")
    for p in paths:
        create_number_csv(p)
        calc_rocs_epoch(p)
