import pickle
from matplotlib import pyplot as plt
import numpy as np
import os

label_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\labels_detailled.pickle'

with open(label_path, 'rb') as f:
    labels = pickle.load(f)

scanners = []
amyloid = []
for k, v in labels.items():
    scanners.append(f"{v['manufacturer']}, {v['model']}")
    amyloid.append(v['label'])

scanners = np.array(scanners)
amyloid = np.array(amyloid)

fig = plt.figure()
plt.title('scanners used in data')
plt.hist(scanners)
plt.savefig(os.path.join(os.path.dirname(label_path), 'scanners'))
plt.title('amyloid status in data')
plt.hist(amyloid)
plt.savefig(os.path.join(os.path.dirname(label_path), 'amyloid'))

print('nice!')