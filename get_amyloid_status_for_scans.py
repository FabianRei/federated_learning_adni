import os
from glob import glob
import pandas as pd
import xml.etree.cElementTree as ET
import numpy as np
import ntpath
import pickle


def get_meta_xml(nii_name, adni_meta, ids, what=('rid', 'examdate')):
    """
    compares the id in the nii file's name with the id in the metadata xml's name.
    This should result in a unique match, as the xml file's ids are unique.
    It parses the corresponding xml file and returns things like RID, EXAMDATE, SCANNER MANUFACTURER and
    SCANNER MODEL.
    Dates use the pandas date format, as it will be compared to that.
    Results are returned in a dict.
    :param nii_name:
    :param adni_meta:
    :param ids:
    :param what:
    :return:
    """
    nii_image_id = nii_name.split('.')[-2].split('_')[-1]
    try:
        meta_file = adni_meta[ids == nii_image_id][0]
    except:
        print(nii_name)
        print(nii_image_id)
        raise Exception
    xml = ET.parse(meta_file)
    result = {}
    for w in what:
        if w == 'rid':
            res = int(xml.find('.//subjectIdentifier').text.split('_')[-1])
            result['rid'] = res
        if w == 'examdate':
            res = xml.find('.//dateAcquired').text
            res = pd.to_datetime(res)
            result['examdate'] = res
        if w == 'manufacturer':
            protocols = xml.findall('.//protocol')
            res = '-1'
            for prot in protocols:
                if prot.attrib['term'] == 'Manufacturer':
                    res = prot.text
            result['manufacturer'] = res
        if w == 'model':
            protocols = xml.findall('.//protocol')
            res = '-1'
            for prot in protocols:
                if prot.attrib['term'] == 'Mfg Model':
                    res = prot.text
            result['model'] = res
        if w == 'is_av45':
            protocols = xml.findall('.//protocol')
            res = '-1'
            for prot in protocols:
                if prot.attrib['term'] == 'Radiopharmaceutical':
                    res = prot.text == '18F-AV45'
            result['is_av45'] = res
        if w == 'img_id':
            result['img_id'] = nii_image_id
    return result


def get_amyloid_label(rid, examdate, berkeley_data, name='-1'):
    """
    Looks into the berkeley study csv file and returns the corresponding label.
    If rid+examdate yield more than one corresponding row, it returns -1, as we then cannot find the
    exact label in the table.
    :param rid:
    :param examdate:
    :return:
    """
    row_idx = ((berkeley_data['EXAMDATE'] == examdate) & (berkeley_data['RID'] == rid))
    if row_idx.sum() > 1:
        print('more than one result for rid & examdate combination in berkeley study data')
        print(f'file name is {name}, rid is {rid}, exam date is {examdate}')
        return -1
    if row_idx.sum() == 0:
        print('no result found for rid & examdate combination in berkeley study data')
        print(f'file name is {name}, rid is {rid}, exam date is {examdate}')
        return -1
    label = berkeley_data['SUMMARYSUVR_WHOLECEREBNORM_1.11CUTOFF'][row_idx]
    return int(label)


def get_labels_from_nifti(nii_paths, berkeley_data, adni_meta, include_notfound=False):
    """
    for a list of paths to nifti, this function extracts the unique image id and then looks up
    rid and exam date at the xml metadata file.
    It then looks into the berkeley av-45 study's table and tries to find the amyloid status based
    on rid and exam date. If there is only one row with that exact rid and exam date, the label is extracted
    and added to a dict that has the file name as key and the label as value. if there is no ore more than one
    corresponding rows with the looked for rid and exam date in the berkeley study table, we either omit that label
    or add -1 for not found.
    :param nii_paths:
    :param berkeley_data:
    :param adni_meta:
    :param include_notfound:
    :return:
    """
    ids = [f.split('.')[-2].split('_')[-1] for f in adni_meta]
    ids = np.array(ids)
    result = {}
    result_detailled = {}
    for nii in nii_paths:
        name = ntpath.basename(nii)[:-4]
        meta_data = get_meta_xml(nii, adni_meta=adni_meta, ids=ids, what=['rid', 'examdate', 'is_av45', 'manufacturer', 'model', 'img_id'])
        if not meta_data['is_av45']:
            print("this ain't the right pet modality!")
        label = get_amyloid_label(meta_data['rid'], meta_data['examdate'], berkeley_data, name=name)
        if include_notfound or label != -1:
            result[name] = label
            meta_data['label'] = label
            meta_data['examdate'] = str(meta_data['examdate']).split(' ')[0]
            result_detailled[name] = meta_data
    return result, result_detailled


# data_folder = r'C:\Users\Fabian\stanford\federated_learning_data'
# berkeley_csv = r'C:\Users\Fabian\stanford\federated_learning_data\UCBERKELEYAV45_04_12_19.csv'
# xml_folder = r'C:\Users\Fabian\stanford\federated_learning_data\full\ADNI_meta'
# nii_folder = r'C:\Users\Fabian\stanford\federated_learning_data\ADNI'
data_folder = '/share/wandell/data/reith/federated_learning/data'
output_folder = '/share/wandell/data/reith/federated_learning'
nii_folder = os.path.join(data_folder, 'data_adni')
xml_folder = os.path.join(data_folder, 'meta_data_adni')
berkeley_csv = os.path.join(data_folder, 'UCBERKELEYAV45_04_12_19.csv')
berkeley_data = pd.read_csv(berkeley_csv, parse_dates=['EXAMDATE'])
# adni_meta = glob(xml_folder + r'\*.xml')
adni_meta = glob(xml_folder + r'/*.xml')
ids = [f.split('.')[-2].split('_')[-1] for f in adni_meta]
adni_meta = np.array(adni_meta)
ids = np.array(ids)
# nii_data = glob(nii_folder + r'\**\*.nii', recursive=True)
nii_data = glob(nii_folder + r'/**/*.nii', recursive=True)

test1 = get_meta_xml(nii_data[0], adni_meta=adni_meta, ids=ids)
print(test1)
test2 = get_meta_xml(nii_data[0], adni_meta=adni_meta, ids=ids, what=['manufacturer', 'model'])
print(test2)
label = get_amyloid_label(rid=test1['rid'], examdate=test1['examdate'], berkeley_data=berkeley_data)
print('nice')
labels, labels_detailled = get_labels_from_nifti(nii_data, berkeley_data, adni_meta)
with open(os.path.join(output_folder, 'labels_plain.pickle'), 'wb') as f:
    pickle.dump(labels, f)
with open(os.path.join(output_folder, 'labels_detailled.pickle'), 'wb') as f:
    pickle.dump(labels_detailled, f)
print('done')