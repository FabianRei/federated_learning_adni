import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd

# label_path = '/share/wandell/data/reith/federated_learning/labels_detailled.pickle'
label_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\labels_detailled.pickle'
with open(label_path, 'rb') as f:
    labels = pickle.load(f)
excel_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\Scanner information.gz.xls'
with open(excel_path, 'rb') as f:
    scan_sheet = pd.read_excel(f)

scanners = []
amyloid = []
rcf = []
img_ids = []
names = []
slices = []
site = []
scan_keys = np.array(scan_sheet['Scanner'])
scanner_keys = np.array([','.join(f.split(',')[:-1]) for f in scan_keys])
scan_numbers = np.array(scan_sheet['Type'])
for k, v in labels.items():
    scanners.append(f"{v['manufacturer']}, {v['model']}")
    amyloid.append(v['label'])
    rcf.append([v['rows'], v['columns'], v['frames']])
    img_ids.append(v['img_id'])
    names.append(k)
    slices.append(v['slices'])
    site.append(v['site'])

site = np.array(site)
names = np.array(names)
slices = np.array(slices)
img_ids = np.array(img_ids)
scanners = np.array(scanners)
amyloid = np.array(amyloid)
rcf = np.array(rcf)
types = []
for s in scanners:
    t = scan_numbers[scanner_keys == s]
    types.append(int(t))

for n, t in zip(names, types):
    labels[n]['scanner_type'] = t
fig = plt.figure()
plt.title('scanners used in data')
plt.hist(scanners)
plt.savefig(os.path.join(os.path.dirname(label_path), 'scanners'))
fig = plt.figure()
plt.title('amyloid status in data')
plt.hist(amyloid)
plt.savefig(os.path.join(os.path.dirname(label_path), 'amyloid'))

print('nice!')