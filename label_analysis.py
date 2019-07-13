import pickle
from matplotlib import pyplot as plt
import numpy as np
import os

label_path = '/share/wandell/data/reith/federated_learning/labels_detailled2.pickle'
label_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\labels_detailled2.pickle'
with open(label_path, 'rb') as f:
    labels = pickle.load(f)

scanners = []
amyloid = []
rcf = []
img_ids = []
names = []
slices = []
for k, v in labels.items():
    scanners.append(f"{v['manufacturer']}, {v['model']}")
    amyloid.append(v['label'])
    rcf.append([v['rows'], v['columns'], v['frames']])
    img_ids.append(v['img_id'])
    names.append(k)
    slices.append(v['slices'])

names = np.array(names)
slices = np.array(slices)
img_ids = np.array(img_ids)
scanners = np.array(scanners)
amyloid = np.array(amyloid)
rcf = np.array(rcf)

fig = plt.figure()
plt.title('scanners used in data')
plt.hist(scanners)
plt.savefig(os.path.join(os.path.dirname(label_path), 'scanners'))
fig = plt.figure()
plt.title('amyloid status in data')
plt.hist(amyloid)
plt.savefig(os.path.join(os.path.dirname(label_path), 'amyloid'))

print('nice!')