import pickle
import numpy as np

p_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\statistics_data_raw.p'

with open(p_path, 'rb') as f:
    p = pickle.load(f)

print(p.keys())
print(p['resnet_50'].keys())
p_50 = p['resnet_50']
p_one = p_50['one_slice']
print(p_one.keys())
preds = []
labels = []
its = []
preds2 = []
print(p_one['binary_pretrained'].keys())
for k, v in p_one['binary_pretrained'].items():
    preds.append(v['prediction_model'])
    its.append(v)
    labels.append(v['label'])
    preds2.append(v['prediction_model_amyloid'])

accs = []
ls = []
ps = []
p2s = []
print('threshold > 0.5')
for p, p2, l in zip(preds, preds2, labels):
    p = (p>0.5).astype(np.float)
    ls.extend(l)
    ps.extend(p)
    p2s.extend(p2)
    accs.append(np.mean(p==l))

ps = np.array(ps)
ls = np.array(ls)
p2s = np.array(p2s)
print(np.mean(accs))
print(np.mean(ps==ls))
print(np.mean(p2s==ls))

accs = []
ls = []
ps = []
print('threshold >= 0.5')
for p, l in zip(preds, labels):
    p = (p>=0.5).astype(np.float)
    ls.extend(l)
    ps.extend(p)
    accs.append(np.mean(p==l))
ps = np.array(ps)
ls = np.array(ls)
print(np.mean(accs))
print(np.mean(ps==ls))