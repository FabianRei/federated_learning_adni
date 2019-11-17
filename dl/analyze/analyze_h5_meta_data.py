import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from dl.data.project_logging import CsvWriter
import h5py


def get_key(key_ids, id):
    for k in key_ids:
        if id == k.split('_')[-1]:
            return k


def get_id(ids, key):
    for i in ids:
        if i == key.spit('_')[-1]:
            return i


# label_path = '/share/wandell/data/reith/federated_learning/labels_detailled.pickle'
h5_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\slice_data_longitudinal.h5'

data = h5py.File(h5_path, 'r')


in_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\xml_labels_detailled_suvr_longitudinal_times_fixed.pickle'

with open(in_path, 'rb') as f:
    data_pickle = pickle.load(f)

ages = []
apoea1 = []
apoea2 = []
dead = []
train_data = []
scan_time = []
sub_id = []
faq_total = []
img_id = []
weight = []
sex = []
label_suvr = []
label_amyloid = []
mmsescore = []
composite_suvr = []


# scan_keys = np.array(scan_sheet['Scanner'])
# scanner_keys = np.array([','.join(f.split(',')[:-1]) for f in scan_keys])
# scan_numbers = np.array(scan_sheet['Type'])
for k in data.keys():
    ages.append(data[k].attrs['age'])
    apoea1.append(data[k].attrs['apoea1'])
    apoea2.append(data[k].attrs['apoea2'])
    dead.append(data[k].attrs['dead'])
    train_data.append(data[k].attrs['train_data'])
    scan_time.append(data_pickle[k]['scan_time'])
    sub_id.append(data[k].attrs['rid'])
    faq_total.append(data[k].attrs['faqtotal'])
    img_id.append(data[k].attrs['img_id'])
    weight.append(data[k].attrs['weight'])
    sex.append(data[k].attrs['sex'])
    label_suvr.append(data[k].attrs['label_suvr'])
    label_amyloid.append(data[k].attrs['label_amyloid'])
    mmsescore.append(data[k].attrs['mmsescore'])
    composite_suvr.append(data[k].attrs['label_0_79_suvr'])


ages = np.array(ages).astype(np.float)
apoea1 = np.array(apoea1).astype(np.float)
apoea2 = np.array(apoea2).astype(np.float)
train_data = np.array(train_data)
scan_time = np.array(scan_time).astype(np.float)
sub_id = np.array(sub_id)
faq_total = np.array(faq_total).astype(np.float)
img_id = np.array(img_id)
weight = np.array(weight).astype(np.float)
sex = np.array(sex)
label_suvr = np.array(label_suvr)
label_amyloid = np.array(label_amyloid)
mmsescore = np.array(mmsescore)
composite_suvr = np.array(composite_suvr)


suvr = label_suvr[~train_data]
time = scan_time[~train_data]
sub = sub_id[~train_data]

delta_s = []
delta_t = []
for s in np.unique(sub):
    su = suvr[sub==s]
    ti = time[sub==s]
    if ti.min() >0:
        if len(ti) > 1:
            ti -= ti.min()
        else:
            continue
    delta_su = su-su[ti.argmin()]
    delta_s.extend(delta_su)
    delta_t.extend(ti)

delta_s = np.array(delta_s)
delta_t = np.array(delta_t)


print('done')
plt.scatter(delta_t[delta_t>0], delta_s[delta_t>0])





# out_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\xml_labels_detailled_suvr_exam_times.pickle'
#
# with open(out_path, 'wb') as f:
#     pickle.dump(labels, f)
