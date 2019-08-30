import pickle
import numpy as np

fpath = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\dist_10_incl_subjects_site_three_slices_dataset_full\epoch_29_pred_labels_train_test_epoch_08-28_21-25.p'

with open(fpath, 'rb') as f:
    data = pickle.load(f)

test = data['test']
