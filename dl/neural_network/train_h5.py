from dl.data.get_dataset import get_dataset
from dl.neural_network.resnet import ResNet50
from dl.neural_network.train_test import train
from dl.data.bin_equal import bin_equal
from datetime import datetime
from dl.data.logging import Logger, CsvWriter
import numpy as np
import torch
from torch import nn
from torch import optim
import sys
import os
import GPUtil
import pickle


def train_h5(h5_path, num_epochs=30, label_names=['label_amyloid'], extra_info='', lr=0.01, decrease_after=10,
             rate_of_decrease=0.1, gpu_device=-1, save_pred_labels=True, test_split=0.2, pretrained=True,
             batch_size=32, binning=-1, regression=False):
    windows_db = False
    if windows_db:
        h5_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\slice_data.h5'

    # chose random when -1, otherwise the selected id
    if gpu_device < 0:
        device_ids = GPUtil.getAvailable(order='first', limit=6, maxLoad=0.1, maxMemory=0.1, excludeID=[],
                                         excludeUUID=[])
        gpu_device = device_ids[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    out_path = os.path.dirname(h5_path)
    time_stamp = datetime.now().strftime('%m-%d_%H-%M')
    # if there are two labels to be fetched, we assume that the first one is trained on and the second one is for
    # testing
    if len(label_names) > 1:
        data, labels, labels2 = get_dataset(h5_path, label_names=label_names)
    else:
        data, labels = get_dataset(h5_path, label_names=label_names)
        # create dummy labels2. not perfect, I guess, but good enough :)
        labels2 = np.ones(len(labels))
    np.random.seed(42)
    shuff_idxs = np.random.permutation(len(data))
    data = data[shuff_idxs]
    labels = labels[shuff_idxs]
    labels2 = labels2[shuff_idxs]
    if binning > 0:
        labels_backup = np.copy(labels)
        labels, break_offs = bin_equal(labels, num_bins=binning)
        with open(os.path.join(out_path, f"original_labels_and_break_offs_{time_stamp}{extra_info}.p"), 'wb') as f:
            pickle.dump({'original_labels': labels_backup, 'break_offs': break_offs}, f)
    # normalize data
    data -= data.mean()
    data /= data.std()

    if windows_db:
        data = data[:100]
        labels = labels[:100]
        labels2 = labels2[:100]
    cutoff = int(len(data)*test_split)
    test_data = data[:cutoff]
    test_labels = labels[:cutoff]
    test_labels2 = labels2[:cutoff]
    train_data = data[cutoff:]
    train_labels = labels[cutoff:]
    train_labels2 = labels2[cutoff:]

    test_data = torch.from_numpy(test_data).type(torch.float32)
    test_labels = torch.from_numpy(test_labels).type(torch.long)
    train_data = torch.from_numpy(train_data).type(torch.float32)
    train_labels = torch.from_numpy(train_labels).type(torch.long)
    num_classes = len(np.unique(labels))

    Net = ResNet50(pretrained=pretrained, num_classes=num_classes)
    Net.cuda()
    criterion = nn.NLLLoss()
    log_path = os.path.join(out_path, f"training_log_{time_stamp}{extra_info}.txt")
    sys.stdout = Logger(log_path)
    csv_path = os.path.join(out_path, f"train_test_accuracy_{time_stamp}{extra_info}.csv")
    header = ['test_acc', 'train_acc', 'train_loss', 'epoch']
    if len(label_names) > 1:
        header.extend(['test_label_acc_train', 'test_label_acc_test'])
    csv_writer = CsvWriter(csv_path, header=header)

    for i in range(num_epochs):
        if i % decrease_after == 0:
            if i > 0:
                lr = lr*rate_of_decrease
            print(f"Trainig for {decrease_after} epochs with a learning rate of {lr}..")
        optimizer = optim.Adam(Net.parameters(), lr=lr)
        train_result = train(batch_size=batch_size, train_data=train_data, train_labels=train_labels, test_data=test_data,
                             test_labels=test_labels, Net=Net, optimizer=optimizer, criterion=criterion,
                             test_interval=1, epochs=1, dim_in='default')
        Net, test_acc, test_pred_label, train_acc, train_loss, train_pred_label = train_result
        if len(label_names) > 1:
            test_label_acc_test = np.mean((test_pred_label[:, 0] >= num_classes/2) == test_labels2)
            test_label_acc_train = np.mean((train_pred_label[:, 0] >= num_classes / 2) == train_labels2)
            csv_writer.write_row(test_acc=test_acc, train_acc=train_acc, train_loss=train_loss, epoch=i,
                                 test_label_acc_train=test_label_acc_train, test_label_acc_test=test_label_acc_test)
            print(f"Amyloid status accuracy is {test_label_acc_test * 100:.2f} percent for train and {test_label_acc_train * 100:.2f} percent for train")
        else:
            csv_writer.write_row(test_acc=test_acc, train_acc=train_acc, train_loss=train_loss, epoch=i)
        if save_pred_labels:
            pickle_fn = os.path.join(out_path, f"epoch_{i}_pred_labels_train_test_epoch_{time_stamp}{extra_info}.p")
            pickle_object = {'train': train_pred_label, 'test': test_pred_label}
            with open(pickle_fn, 'wb') as f:
                pickle.dump(pickle_object, f)
        print(f"Test accuracy is {test_acc * 100:.2f} percent")


if __name__ == '__main__':
    train_h5(r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\slice_data.h5', label_names=['label_suvr', 'label_amyloid'], binning=20,
             num_epochs=3)




