from dl.data.get_dataset import get_dataset
from dl.neural_network.resnet import ResNet50, ResNet50Reg
from dl.neural_network.train_test import train
from dl.neural_network.train_test_regression import train_reg
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
             batch_size=32, binning=-1, regression=False, include_subject_ids=True):
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
        if include_subject_ids:
            data, labels, labels2, s_ids = get_dataset(h5_path, label_names=label_names, include_subjects=True)
        else:
            data, labels, labels2 = get_dataset(h5_path, label_names=label_names)
    else:
        if include_subject_ids:
            data, labels, s_ids = get_dataset(h5_path, label_names=label_names, include_subjects=True)
        else:
            data, labels = get_dataset(h5_path, label_names=label_names)        # create dummy labels2. not perfect, I guess, but good enough :)
        # turns out that I actually don't need the amyloid status label, as the suvr value is obviously sufficient to
        # infer amyloid status. Still keeping it. Why, you ask? Because I can.
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
    # We try to not have the same subject in train and test set. To do so, we iteratively assign from subjects with a
    # high number of scans (max is 5) to subjects with a low number of scans (1) to train and test set. If the
    # train-test split is 4 to 1, we assign 4 subjects to train and one to test in each iteration etc. etc.
    if include_subject_ids:
        ratio = int(np.round((1-test_split)/test_split))
        n, c = np.unique(s_ids, return_counts=True)
        sort_unique = np.argsort(n)[::-1]
        n = list(n[sort_unique])
        c = list(c[sort_unique])
        test_size = int(len(data) * test_split)
        train_size = len(data)-test_size
        test_idxs = np.zeros(len(data)).astype(np.int)
        train_idxs = np.zeros(len(data)).astype(np.int)
        while True:
            try:
                for _ in range(int(np.floor(ratio/2))):
                    if c[0]+np.sum(train_idxs) <= train_size:
                        cc = c.pop(0)
                        nnn = n.pop(0)
                        train_idxs[s_ids == nnn] = 1
                if c[0] + np.sum(test_idxs) <= test_size:
                    cc = c.pop(0)
                    nnn = n.pop(0)
                    test_idxs[s_ids == nnn] = 1
                for _ in range(int(np.ceil(ratio/2))):
                    if c[0]+np.sum(train_idxs) <= train_size:
                        cc = c.pop(0)
                        nnn = n.pop(0)
                        train_idxs[s_ids == nnn] = 1
                if np.sum(test_idxs) == test_size and np.sum(train_idxs) == train_size:
                    break
            except:
                break
        print('data split!')
        print(f"Test set, goal of {test_size}, got {np.sum(test_idxs)}")
        print(f"Trai set, goal of {train_size}, got {np.sum(train_idxs)}")
        train_idxs = np.where(train_idxs)[0]
        test_idxs = np.where(test_idxs)[0]
        test_data = data[test_idxs]
        test_labels = labels[test_idxs]
        test_labels2 = labels2[test_idxs]
        train_data = data[train_idxs]
        train_labels = labels[train_idxs]
        train_labels2 = labels2[train_idxs]
    else:
        cutoff = int(len(data) * test_split)
        test_data = data[:cutoff]
        test_labels = labels[:cutoff]
        test_labels2 = labels2[:cutoff]
        train_data = data[cutoff:]
        train_labels = labels[cutoff:]
        train_labels2 = labels2[cutoff:]

    train_data = torch.from_numpy(train_data).type(torch.float32)
    test_data = torch.from_numpy(test_data).type(torch.float32)
    if len(train_data.shape) > 3:
        # we then have an extra dimension with channels
        train_data = train_data.permute(0, 3, 1, 2)
        test_data = test_data.permute(0, 3, 1, 2)
    num_classes = len(np.unique(labels))

    if regression:
        test_labels = torch.from_numpy(test_labels).type(torch.float32)
        train_labels = torch.from_numpy(train_labels).type(torch.float32)
        Net = ResNet50Reg(pretrained=pretrained, num_classes=1)
        criterion = nn.MSELoss()
        train_func = train_reg
    else:
        test_labels = torch.from_numpy(test_labels).type(torch.long)
        train_labels = torch.from_numpy(train_labels).type(torch.long)
        Net = ResNet50(pretrained=pretrained, num_classes=num_classes)
        criterion = nn.NLLLoss()
        train_func = train

    Net.cuda()
    log_path = os.path.join(out_path, f"training_log_{time_stamp}{extra_info}.txt")
    sys.stdout = Logger(log_path)
    standard_info = f"_lr_{str(lr).replace('.', '_')}_{'pretrained' if pretrained else 'non_pretrained'}_{'reg_' if regression else ''}{str(binning)+'bins' if binning>0 else ''}{num_epochs}epochs_rod_{str(rate_of_decrease).replace('.', '_')}_da_{decrease_after}"
    csv_path = os.path.join(out_path, f"train_test_accuracy_{time_stamp}{extra_info}{standard_info}.csv")
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
        train_result = train_func(batch_size=batch_size, train_data=train_data, train_labels=train_labels, test_data=test_data,
                                  test_labels=test_labels, Net=Net, optimizer=optimizer, criterion=criterion,
                                  test_interval=1, epochs=1, dim_in='default')
        Net, test_acc, test_pred_label, train_acc, train_loss, train_pred_label = train_result
        if len(label_names) > 1 and not regression:
            test_label_acc_test = np.mean((test_pred_label[:, 0] >= num_classes/2) == test_labels2)
            test_label_acc_train = np.mean((train_pred_label[:, 0] >= num_classes / 2) == (train_pred_label[:, 1] >= num_classes / 2))
            csv_writer.write_row(test_acc=test_acc, train_acc=train_acc, train_loss=train_loss, epoch=i,
                                 test_label_acc_train=test_label_acc_train, test_label_acc_test=test_label_acc_test)
            print(f"Amyloid status accuracy is {test_label_acc_test * 100:.2f} percent for test and {test_label_acc_train * 100:.2f} percent for train")
        elif len(label_names) > 1 and regression:
            test_label_acc_test = np.mean((test_pred_label[:, 0] >= 1.11) == (test_pred_label[:, 1] >= 1.11))
            test_label_acc_train = np.mean((train_pred_label[:, 0] >= 1.11) == (train_pred_label[:, 1] >= 1.11))
            csv_writer.write_row(test_acc=test_acc, train_acc=train_acc, train_loss=train_loss, epoch=i,
                                 test_label_acc_train=test_label_acc_train, test_label_acc_test=test_label_acc_test)
            print(f"Amyloid status accuracy is {test_label_acc_test * 100:.2f} percent for test and {test_label_acc_train * 100:.2f} percent for train")
        else:
            csv_writer.write_row(test_acc=test_acc, train_acc=train_acc, train_loss=train_loss, epoch=i)
        if save_pred_labels:
            pickle_fn = os.path.join(out_path, f"epoch_{i}_pred_labels_train_test_epoch_{time_stamp}{extra_info}.p")
            pickle_object = {'train': train_pred_label, 'test': test_pred_label}
            with open(pickle_fn, 'wb') as f:
                pickle.dump(pickle_object, f)
        print(f"Test accuracy is {test_acc * 100:.2f} percent")


if __name__ == '__main__':
    train_h5(r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\slice_data_subj.h5', label_names=['label_suvr', 'label_amyloid'], binning=-1,
             num_epochs=10, regression=False, lr=0.0001)




