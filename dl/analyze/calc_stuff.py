import csv
from visualize.calc_roc import cutoff_youdens_j_tt, calc_sensitivity, calc_specificity, calc_npv, calc_ppv, calc_roc_auc
import pandas as pd
import numpy as np

def get_csv_column(csv_path, col_name, sort_by=None, exclude_from=None):
    df = pd.read_csv(csv_path, delimiter=',')
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



fpath = r'C:\Users\Fabian\Desktop\clinical_analysis.csv'

reader1 = get_csv_column(fpath, 'Reader 1')
reader2 = get_csv_column(fpath, 'Reader 2')
reader3 = get_csv_column(fpath, 'Reader 3')
prediction_amyloid = get_csv_column(fpath, 'prediction_amyloid')
label_amyloid = get_csv_column(fpath, 'label_amyloid')
allreaders = np.round(np.mean([reader1, reader2, reader3, prediction_amyloid, prediction_amyloid], axis=0))

res_dict = {}
res_dict['metric'] = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
for k,v in {'ResNet-50':prediction_amyloid, 'Reader 1':reader1, 'Reader 2':reader2, 'Reader 3':reader3, 'All readers':allreaders}.items():
    # acc, sens, spec, ppv, npv
    tt = np.ones((len(reader1), 2))
    tt[:, 0] = v
    tt[:, 1] = label_amyloid
    res = []
    res.append(np.mean(v==label_amyloid))
    res.append(calc_sensitivity(tt))
    res.append(calc_specificity(tt))
    res.append(calc_ppv(tt))
    res.append(calc_npv(tt))
    res_dict[k] = res

res_df = pd.DataFrame.from_dict(res_dict)
res_df.to_csv( r'C:\Users\Fabian\Desktop\clinical_analysis_res.csv')

print('nice')