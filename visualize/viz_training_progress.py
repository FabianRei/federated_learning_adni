import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from glob import glob
import re
from fnmatch import fnmatch

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


def find_all_identifiers(folder, file_ending='.csv', within_file_pattern=''):
    identifiers = []
    files = [os.path.basename(f) for f in glob(os.path.join(folder, f'*{file_ending}'))]
    for file in files:
        ids = re.findall(rf'\d+-\d+_\d+-\d+.*{within_file_pattern}.*.csv', file)
        ids = [id[:-4] for id in ids]
        identifiers.extend(ids)
    return identifiers


if __name__ == '__main__':
    fpath = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments'
    paths = glob(fpath + r"\*site*")
    for p in paths:

        # fpath = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\more_one_slice_dataset'
        # fpath = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\incl_subjects_one_slices_dataset_full'
        # fpath = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\dist_40_incl_subjects_site_three_slices_dataset_full'
        # identifier = r'07-30_09-40_pretrain_normalizeData'
        # viz_training(fpath, identifier)
        fpath = p
        ids = find_all_identifiers(fpath)
        for ident in ids:
            viz_training(fpath, ident)
        ids_bin = find_all_identifiers(fpath, file_ending='.csv', within_file_pattern='bin')
        ids_reg = find_all_identifiers(fpath, file_ending='.csv', within_file_pattern='reg')
        for ident in ids_reg:
            viz_training(fpath, ident, train_acc='test_label_acc_train', test_acc='test_label_acc_test', fname_addition='_reg_test')
        for ident in ids_bin:
            viz_training(fpath, ident, train_acc='test_label_acc_train', test_acc='test_label_acc_test',
                         fname_addition='_bin_test')

