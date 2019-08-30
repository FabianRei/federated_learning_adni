import pickle
import numpy as np

fpath = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\dist_10_incl_subjects_site_three_slices_dataset_full\epoch_29_pred_labels_train_test_epoch_08-28_21-25.p'
fpath2 = r''
with open(fpath, 'rb') as f:
    data = pickle.load(f)

test = data['test']
tt = (test>1.11).astype(np.int)


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

