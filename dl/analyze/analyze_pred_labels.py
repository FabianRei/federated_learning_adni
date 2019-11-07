import numpy as np
import pickle
from glob import glob
import os
import csv


def get_contrast(path):
    part2 = path.split('_')[-8]
    part1 = path.split('_')[-9]
    contrast = part1 + '.' + part2
    contrast = float(contrast)
    return contrast


def sort_paths(paths):
    paths.sort(key=lambda x: int(x.split('_')[-8]))
    return paths


def write_csv_row(resultCSV, testAcc, accOptimal, d1, d2, dataContrast, nn_dprime):
    file_exists = os.path.isfile(resultCSV)
    with open(resultCSV, 'a') as csvfile:
        headers = ['ResNet_accuracy', 'optimal_observer_accuracy', 'theoretical_d_index', 'optimal_observer_d_index',
                   'contrast', 'nn_dprime']
        writer = csv.DictWriter(csvfile, delimiter=';', lineterminator='\n', fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow({'ResNet_accuracy': testAcc, 'optimal_observer_accuracy': accOptimal, 'theoretical_d_index': d1,
                         'optimal_observer_d_index': d2, 'contrast': dataContrast,
                         'nn_dprime': nn_dprime})


def write_csv_svm(resultCSV, svm_accuracy, dprime_accuracy, contrast, samples_used=1000):
    file_exists = os.path.isfile(resultCSV)
    with open(resultCSV, 'a') as csvfile:
        headers = ['svm_accuracy', 'dprime_accuracy', 'contrast', 'samples_used']
        writer = csv.DictWriter(csvfile, delimiter=';', lineterminator='\n', fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow(
            {'svm_accuracy': svm_accuracy, 'dprime_accuracy': dprime_accuracy, 'contrast': contrast, 'samples_used': samples_used})


def calculate_values(oo_data, nn_data, svm_data, out_path, adjust_imbalance=True):
    nn_data = sort_paths(nn_data)
    oo_data = sort_paths(oo_data)
    svm_data = sort_paths(svm_data)
    csv_path = os.path.join(out_path, 'results.csv')
    if os.path.exists(csv_path):
        os.replace(csv_path, os.path.join(out_path, 'results_old.csv'))
    csv_svm = os.path.join(out_path, 'svm_results_seeded.csv')
    if os.path.exists(csv_svm):
        os.replace(csv_svm, os.path.join(out_path, 'svm_results_seeded_old.csv'))
    for o, n, s in zip(oo_data, nn_data, svm_data):
        with open(o, 'rb') as f:
            oo = pickle.load(f)
        with open(n, 'rb') as f:
            nn = pickle.load(f)
        with open(s, 'rb') as f:
            svm = pickle.load(f)
        if not (get_contrast(o) == get_contrast(n) == get_contrast(s)):
            print('ERROR, contrast not the same..')
            raise AttributeError
        contrast = get_contrast(o)
        oo_acc = np.mean(oo[:,0] == oo[:,1])
        nn_acc = np.mean(nn[:,0] == nn[:,1])
        oo_dprime = calculate_dprime(oo, adjust_imbalance=adjust_imbalance)
        nn_dprime = calculate_dprime(nn, adjust_imbalance=adjust_imbalance)
        svm_dprime = calculate_dprime(svm, adjust_imbalance=adjust_imbalance)
        svm_acc = np.mean(svm[:,0] == svm[:,1])
        write_csv_row(csv_path, nn_acc, oo_acc, -1, oo_dprime, contrast, nn_dprime)
        write_csv_svm(csv_svm, svm_acc, svm_dprime, contrast)


# create result csv files based on pickle prediction-labels
fp1 = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds\seed_11\one_slice\epoch_8_pred_labels_train_test_epoch_09-17_17-57_lr_0_001_pretrained_reg_30epochs_rod_0_1_da_10.p'
fp2 = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds\seed_11\one_slice\epoch_9_pred_labels_train_test_epoch_09-17_17-57_lr_0_001_pretrained_reg_30epochs_rod_0_1_da_10.p'
fp1 = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds_lower_lr_imgnet_fixed\seed_11\one_slice\epoch_29_pred_labels_train_test_epoch_09-23_17-50_lr_0_0001_pretrained_reg_30epochs_rod_0_1_da_10.p'
with open(fp1, 'rb') as f:
    res1 = pickle.load(f)
with open(fp2, 'rb') as f:
    res2 = pickle.load(f)
test1 = res1['test']
t1 = (test1[:,0]-test1[:,1])**2
print(np.mean(t1))
te1 = test1 > 1.11
print(np.mean(te1[:,0]==te1[:,1]))
print('nice')
test = res2['test']
t = (test[:,0]-test[:,1])**2
print(np.mean(t))
te = test > 1.11
print(np.mean(te[:,0]==te[:,1]))
print('nice')




