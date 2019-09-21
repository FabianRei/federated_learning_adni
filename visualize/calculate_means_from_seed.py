import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from glob import glob
import re
from fnmatch import fnmatch
from fnmatch import filter
import csv


def get_csv_column(csv_path, col_name, sort_by=None, exclude_from=None):
    df = pd.read_csv(csv_path, delimiter=';')
    col = df[col_name].tolist()
    col = np.array(col)
    if sort_by is not None:
        sort_val = get_csv_column(csv_path, sort_by)
        sort_idxs = np.argsort(sort_val)
        col = col[sort_idxs]
    if exclude_from is not None:
        sort_val = sort_val[sort_idxs]
        col = col[sort_val >= exclude_from]
    return col


def viz_training(folder, identifier='', sort_by='epoch', title='default', train_acc='train_acc', test_acc='test_acc',
                 fname_addition=""):
    csv_path = glob(os.path.join(folder, f"*{identifier}*.csv"))[0]
    fname = f"viz_of_{identifier}{fname_addition}.png"
    fig = plt.figure()
    plt.xlabel('Epochs')
    if fnmatch(identifier, '*reg*') and fname_addition == "":
        plt.ylabel("Loss (MSE)")
    else:
        plt.ylabel('Accuracy')
    # num = folder_path.split('_')[-1]
    if title == 'default':
        title = f"Training progression for {identifier}{fname_addition}"
    plt.title(title)
    plt.grid(which='both')
    train_acc = get_csv_column(csv_path, train_acc, sort_by=sort_by)
    test_acc = get_csv_column(csv_path, test_acc, sort_by=sort_by)
    epochs = get_csv_column(csv_path, 'epoch', sort_by=sort_by)
    epochs += 1
    plt.plot(epochs, train_acc, label='Train data')
    plt.plot(epochs, test_acc, label='Test data')

    plt.legend(frameon=True, loc='upper left', fontsize='small')
    fig.savefig(os.path.join(folder, f'{fname}.png'), dpi=200)
    # fig.show()
    print('done!')


def viz_training_data(out_folder, train_acc, test_acc, epochs, identifier='', title='default', fname_addition=''):
    fname = f"viz_of_{identifier}{fname_addition}.png"
    fig = plt.figure()
    plt.xlabel('Epochs')
    if fnmatch(identifier, '*reg*') and fname_addition == "":
        plt.ylabel("Loss (MSE)")
    else:
        plt.ylabel('Accuracy')
    # num = folder_path.split('_')[-1]
    if title == 'default':
        title = f"Training progression for {identifier}{fname_addition}"
    plt.title(title)
    plt.grid(which='both')
    plt.plot(epochs, train_acc, label='Train data')
    plt.plot(epochs, test_acc, label='Test data')
    plt.legend(frameon=True, loc='upper left', fontsize='small')
    fig.savefig(os.path.join(out_folder, f'{fname}.png'), dpi=200)
    # fig.show()
    print('done!')


def find_all_identifiers(folder, file_ending='.csv', within_file_pattern=''):
    identifiers = []
    files = [os.path.basename(f) for f in glob(os.path.join(folder, f'*{file_ending}'))]
    for file in files:
        ids = re.findall(rf'\d+-\d+_\d+-\d+.*{within_file_pattern}.*.csv', file)
        ids = [id[:-4] for id in ids]
        identifiers.extend(ids)
    return identifiers


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


def get_spec_mean_std(csv_files, col_name, sort_by='epoch', include_raw=True):
    res_accs = []
    for f in csv_files:
        res_acc = get_csv_column(f, col_name, sort_by=sort_by)
        res_accs.append(res_acc)
    res_accs = np.stack(res_accs)
    mean_res = np.mean(res_accs, axis=0)
    std_res = np.std(res_accs, axis=0)
    if include_raw:
        return mean_res, std_res, res_accs
    else:
        return mean_res, std_res


def get_mean_std(csv_files, cols=['train_acc', 'test_acc'], sort_by='epoch'):
    # assumes the same epochs for all csv files
    epochs = get_csv_column(csv_files[0], 'epoch', sort_by=sort_by) + 1
    result_dict = {'epochs': epochs}
    raw_dict = {}
    for col in cols:
        mean_res, std_res, raw_res = get_spec_mean_std(csv_files, col)
        result_dict['mean_' + col] = mean_res
        result_dict['std_' + col] = std_res
        raw_dict[col] = raw_res
    return result_dict, raw_dict


def write_args2csv(dict, id, out_path):
    csv_path = os.path.join(out_path, f'{id}.csv')
    file_exists = os.path.isfile(csv_path)
    if file_exists:
        os.remove(csv_path)
    headers = [h for h in dict.keys()]
    with open(csv_path, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
        writer.writeheader()
        for i in range(len(dict[headers[0]])):
            row = {k: v[i] for k, v in dict.items()}
            writer.writerow(row)


def write_raw2csv(dict, id, out_path):
    headers = list(range(10, 15))
    for k, v in dict.items():
        csv_path = os.path.join(out_path, f'{id}_{k}_raw.csv')
        file_exists = os.path.isfile(csv_path)
        if file_exists:
            os.remove(csv_path)
        with open(csv_path, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
            writer.writeheader()
            for i in range(len(v[0])):
                row = {k: val[i] for k, val in zip(headers, v)}
                writer.writerow(row)


if __name__ == '__main__':
    super_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds_10-90'
    super_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\seeds_lower_lr'
    fpaths = glob(os.path.join(super_path, '*seed*'))
    dist_10 = glob(rf'{super_path}\*seed*\dist_10\*.csv')
    one_slice = glob(rf'{super_path}\*seed*\one_slice\*.csv')
    three_slice = glob(rf'{super_path}\*seed*\three_slice\*.csv')
    nine_slice = glob(rf'{super_path}\*seed*\slices_10-90\*.csv')
    csv_dict = {'dist_10': dist_10, 'one_slice': one_slice, 'three_slice': three_slice, 'nine_slice': nine_slice}

    for key, val in csv_dict.items():
        if not val:
            continue
        out_path = os.path.join(super_path, 'summary', key)
        os.makedirs(out_path, exist_ok=True)
        # case binary pretrained
        case_id = 'binary_pretrained'
        csv_files = filter(val, '*1_pretrained_3*')
        if csv_files:
            columns = ['test_acc', 'train_acc', 'train_loss', 'test_label_acc_train', 'test_label_acc_test']
            res, raw_res = get_mean_std(csv_files, columns)
            viz_training_data(out_path, res['mean_train_acc'], res['mean_test_acc'], res['epochs'], identifier=case_id)
            write_args2csv(res, case_id, out_path)
            write_raw2csv(raw_res, case_id, out_path)
        # case binary non pretrained
        case_id = 'binary_non_pretrained'
        csv_files = filter(val, '*non_pretrained_3*')
        if csv_files:
            columns = ['test_acc', 'train_acc', 'train_loss', 'test_label_acc_train', 'test_label_acc_test']
            res, raw_res = get_mean_std(csv_files, columns)
            viz_training_data(out_path, res['mean_train_acc'], res['mean_test_acc'], res['epochs'], identifier=case_id)
            write_args2csv(res, case_id, out_path)
            write_raw2csv(raw_res, case_id, out_path)
        # case pretrained regression
        case_id = 'pretrained_regression'
        csv_files = filter(val, '*1_pretrained_reg*')
        if csv_files:
            columns = ['test_acc', 'train_acc', 'train_loss', 'test_label_acc_train', 'test_label_acc_test']
            res, raw_res = get_mean_std(csv_files, columns)
            viz_training_data(out_path, res['mean_test_label_acc_train'], res['mean_test_label_acc_test'], res['epochs'],
                              identifier=case_id)
            write_args2csv(res, case_id, out_path)
            write_raw2csv(raw_res, case_id, out_path)
        # case non retrained regression
        case_id = 'non_pretrained_regression'
        if csv_files:
            csv_files = filter(val, '*non_pretrained_reg*')
            columns = ['test_acc', 'train_acc', 'train_loss', 'test_label_acc_train', 'test_label_acc_test']
            res, raw_res = get_mean_std(csv_files, columns)
            viz_training_data(out_path, res['mean_test_label_acc_train'], res['mean_test_label_acc_test'], res['epochs'],
                              identifier=case_id)
            write_args2csv(res, case_id, out_path)
            write_raw2csv(raw_res, case_id, out_path)
