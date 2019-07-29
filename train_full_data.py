from dl.data.get_dataset import get_dataset
from dl.neural_network.resnet import ResNet50
import numpy as np
import torch

windows_db = True
test_split = 0.2
if windows_db:
    h5_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\slice_data.h5'
else:
    h5_path = ''
data, labels = get_dataset(h5_path)
np.random.seed(42)
shuff_idxs = np.random.permutation(len(data))
data = data[shuff_idxs]
labels = labels[shuff_idxs]
cutoff = int(len(data)*0.2)
test_data = data[:cutoff]
test_labels = data[:cutoff]
train_data = data[cutoff:]
train_labels = data[cutoff:]

test_data = torch.from_numpy(test_data).type(torch.float32)
test_labels = torch.from_numpy(test_labels.astype(np.long))
train_data = torch.from_numpy(train_data).type(torch.float32)
train_labels = torch.from_numpy(train_labels.astype(np.long))

ResNet = ResNet50(pretrained=True)




