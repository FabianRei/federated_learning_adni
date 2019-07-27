import numpy as np
import os
from glob import glob
import pickle
import h5py
import nibabel as nib

def get_fname(path_name):
    return os.path.splitext(os.path.basename(path_name))[0]


windows_db = False

if windows_db:
    fpath = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\trial_sample'
    prefix = '\\\\?\\'
    outpath = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\one_slice_dataset'
    os.makedirs(outpath, exist_ok=True)
else:
    fpath = '/share/wandell/data/reith/federated_learning/data'
    prefix = ''
    outpath = '/share/wandell/data/reith/federated_learning/data/one_slice_dataset'
    os.makedirs(outpath, exist_ok=True)

xml_path = os.path.join(fpath, 'xml')
nifti_path = os.path.join(fpath, 'nifti')
pickle_path = os.path.join(fpath, 'labels_detailled_suvr.pickle')

with open(pickle_path, 'rb') as p:
    pdata = pickle.load(p)

pickle_fnames = list(pdata.keys())
nifti_files = glob(f'{nifti_path}/**/*.nii', recursive=True)

sizes = []
h5_file = h5py.File(os.path.join(outpath, 'slice_data.h5'))
labels_amyloid = {}
labels_suvr = {}
for f in nifti_files:
    basename = get_fname(f)
    if basename in pickle_fnames:
        img = nib.load(prefix + f)
        sizes.append(img.shape)
        arr = img.get_fdata()
        arr = arr[:, :, 50, 0]
        h5_file.create_dataset(basename, data=arr)
        labels_amyloid[basename] = pdata[basename]['label']
        labels_suvr[basename] = pdata[basename]['label_suvr']

with open(os.path.join(outpath, 'labels_amyloid.pickle'), 'rb') as f:
    pickle.dump(labels_amyloid, f)

with open(os.path.join(outpath, 'labels_suvr.pickle'), 'rb') as f:
    pickle.dump(labels_suvr, f)
h5_file.close()
print('done!')

