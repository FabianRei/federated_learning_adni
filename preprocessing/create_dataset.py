import numpy as np
import os
from glob import glob
import pickle
import h5py
import nibabel as nib

def get_fname(path_name):
    return os.path.splitext(os.path.basename(path_name))[0]


windows_db = True

if windows_db:
    fpath = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\trial_sample'
    prefix = '\\\\?\\'
    outpath = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\dist_20_incl_subjects_site_three_slices_dataset'
    os.makedirs(outpath, exist_ok=True)
else:
    fpath = '/scratch/reith/fl/data'
    prefix = ''
    outpath = '/scratch/reith/fl/experiments/slices_27_data'
    os.makedirs(outpath, exist_ok=True)

xml_path = os.path.join(fpath, 'xml')
nifti_path = os.path.join(fpath, 'nifti')
pickle_path = os.path.join(fpath, 'labels_detailled_suvr.pickle')

with open(pickle_path, 'rb') as p:
    pdata = pickle.load(p)

pickle_fnames = list(pdata.keys())
nifti_files = glob(f'{nifti_path}/**/*.nii', recursive=True)

sizes = []
h5_file = h5py.File(os.path.join(outpath, 'slice_data_subj.h5'), 'w')
# labels_amyloid = {}
# labels_suvr = {}
write_file = open(os.path.join(outpath, 'faulty_nii_files.txt'), 'w')
for i, f in enumerate(nifti_files):
    basename = get_fname(f)
    if basename in pickle_fnames:
        try:
            img = nib.load(prefix + f)
            sizes.append(img.shape)
            arr = img.get_fdata()
            # arr = arr[:, :, 50, 0]
            # arr = arr[:, :, [10, 20, 30, 40, 50, 60, 70, 80, 90], 0]
            slices = np.linspace(0, 93, num=27, dtype=np.int)
            arr = arr[:, :, slices, 0]
            h5_file.create_dataset(basename, data=arr)
            h5_file[basename].attrs['label_amyloid'] = pdata[basename]['label']
            h5_file[basename].attrs['label_suvr'] = pdata[basename]['label_suvr']
            h5_file[basename].attrs['rid'] = pdata[basename]['rid']
            h5_file[basename].attrs['site'] = pdata[basename]['site']
        except Exception as e:
            print(f'{basename} sucks, error is: {e}')
            write_file.write(f'{basename}, {e} \n')
            continue
        # labels_amyloid[basename] = pdata[basename]['label']
        # labels_suvr[basename] = pdata[basename]['label_suvr']
        print(f"{i*100/len(nifti_files):.2f}%. I did {basename}")


write_file.close()
# with open(os.path.join(outpath, 'labels_amyloid.pickle'), 'wb') as f:
#     pickle.dump(labels_amyloid, f)
#
# with open(os.path.join(outpath, 'labels_suvr.pickle'), 'wb') as f:
#     pickle.dump(labels_suvr, f)
h5_file.close()
print('done!')

