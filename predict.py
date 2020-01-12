import torch
import nibabel
import numpy as np
from torch.autograd import Variable


model_path = r'model.pth'
nii_file = r'data.nii'

nii = nibabel.load(nii_file)
nii = nii.get_fdata()
# get the three slices used for prediction
nii = nii[:, :, [40, 50, 60], 0]

# normalization steps used in training and testing. Different data might differ
# channel wise subtraction of mean and division by std
dataset_mean = [0.56887938, 0.56002837, 0.54177473]
dataset_std = [0.72242739, 0.74183118, 0.76492291]
nii -= dataset_mean
nii /= dataset_std

# squeeze between 0 and 1. Imagenet means are then substracted by ResNet class itself
# ImageNet weights require data to be between 0 and 1 (before specific normalization)
dataset_max = 8.876033229409499
dataset_min = -1.5854167773973693
nii -= dataset_min
nii /= (dataset_max-dataset_min)

# get into correct shape
nii = np.transpose(nii, [2,0,1])
nii = np.expand_dims(nii, 0)

# to torch format
nii = torch.tensor(nii)
nii = nii.float()


model = torch.load(model_path)
model.to_cpu()
model.eval()

# model_input gets modified, as reference given
model_in = nii.clone()
result = model(model_in)
print(result)
print('nice')