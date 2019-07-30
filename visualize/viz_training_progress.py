import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from glob import glob


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


def viz_training(folder, identifier='', sort_by='epoch', title='default'):
    csv_path = glob(os.path.join(folder, f"*{identifier}*.csv"))[0]
    fname = f"viz_of_{identifier}.png"
    fig = plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # num = folder_path.split('_')[-1]
    if title == 'default':
        title = f"Training progression for {identifier}"
    plt.title(title)
    plt.grid(which='both')
    train_acc = get_csv_column(csv_path, 'train_acc', sort_by=sort_by)
    test_acc = get_csv_column(csv_path, 'test_acc', sort_by=sort_by)
    epochs = get_csv_column(csv_path, 'epoch', sort_by=sort_by)

    plt.plot(epochs, train_acc, label='Accuracy train data')
    plt.plot(epochs, test_acc, label='Accuracy test data')

    plt.legend(frameon=True, loc='upper left', fontsize='small')
    fig.savefig(os.path.join(folder, f'{fname}.png'), dpi=200)
    # fig.show()
    print('done!')

if __name__ == '__main__':
    fpath = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\one_slice_dataset'
    identifier = r'07-30_09-40_pretrain_normalizeData'
    viz_training(fpath, identifier)
