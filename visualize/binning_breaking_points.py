from glob import glob
import re
import pickle

# f_paths = glob(f'{folder}\*epoch_{epoch}_pred_labels*.p')
# f_paths = [p for p in f_paths if re.match(r'.*reg.*|.*20bins.*', p) is None]
fpaths = [r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\incl_subjects_site_three_slices_dataset_full\original_labels_and_break_offs_09-01_13-45_lr_0_001_non_pretrained_20bins30epochs_rod_0_1_da_10.p']
for fp in fpaths:
    with open(fp, 'rb') as f:
        tt = pickle.load(f)
        tt = tt['test']