import numpy as np
import os
from glob import glob
import pickle
import nibabel as nib

def get_fname(path_name):
    return os.path.splitext(os.path.basename(path_name))[0]


windows_db = False

if windows_db:
    fpath = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\trial_sample'
    prefix = '\\\\?\\'
else:
    fpath = '/share/wandell/data/reith/federated_learning/data'
    prefix = ''
xml_path = os.path.join(fpath, 'xml')
nifti_path = os.path.join(fpath, 'nifti')
pickle_path = os.path.join(fpath, 'labels_detailled_suvr.pickle')

with open(pickle_path, 'rb') as p:
    pdata = pickle.load(p)

pickle_fnames = list(pdata.keys())
nifti_files = glob(f'{nifti_path}/**/*.nii', recursive=True)

sizes = []

for f in nifti_files:
    basename = get_fname(f)
    if basename in pickle_fnames:
        img = nib.load(prefix + f)
        sizes.append(img.shape)

for s in sizes:
    if s != (160, 160, 96, 1):
        print(s)


print('done!')

