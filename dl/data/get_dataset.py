import h5py
import numpy as np


def get_dataset(h5_path, label_names=['label_amyloid'], limit=-1):
    data = h5py.File(h5_path, 'r')
    arrs = []
    labels = []
    for i in range(len(label_names)):
        labels.append([])
    for k in data.keys():
        arrs.append(data[k][()])
        for i, l in enumerate(label_names):
            labels[i].append(data[k].attrs[l])
    labels = [np.array(l) for l in labels]
    arrs = np.array(arrs)
    return (arrs, *labels)


if __name__ == '__main__':
    fp = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\slice_data.h5'
    dset, labels = get_dataset(fp)
    dset, l1, l2 = get_dataset(fp, label_names=['label_amyloid', 'label_suvr'])
    print('done')
