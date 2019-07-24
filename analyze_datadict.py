import pandas as pd
import numpy as np
from fnmatch import fnmatch

fpath = r'C:\Users\Fabian\Downloads\DATADIC.csv'

data = pd.read_csv(fpath)
fld = np.array(data['FLDNAME'])
# fld = np.array(data['TEXT'])
code = np.array(data['CODE'])
# code = np.array(data['TEXT'])
fl_unique = np.unique([f for f in fld if fnmatch(f, '*site*')])

for fl in fl_unique:
    match_idxs = fld == fl
    matches = code[match_idxs]
    for m in matches:
        try:
            if not np.isnan(m):
                print(m)
        except:
            print('error thrown', m)
print('nice')