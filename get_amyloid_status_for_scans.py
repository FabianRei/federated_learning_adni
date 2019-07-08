import os
from glob import glob
import pandas as pd
import xml.etree.cElementTree as ET
import numpy as np
import ntpath


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
    meta_file = adni_meta[ids == nii_image_id][0]
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
    for nii in nii_paths:
        name = ntpath.basename(nii)[:-4]
        meta_data = get_meta_xml(nii, adni_meta=adni_meta, ids=ids, what=['rid', 'examdate'])
        label = get_amyloid_label(meta_data['rid'], meta_data['examdate'], berkeley_data, name=name)
        if include_notfound or label != -1:
            result[name] = label
    return result



berkeley_csv = r'C:\Users\Fabian\stanford\federated_learning_data\UCBERKELEYAV45_04_12_19.csv'
berkeley_data = pd.read_csv(berkeley_csv, parse_dates=['EXAMDATE'])
adni_meta = glob(r'C:\Users\Fabian\stanford\federated_learning_data\full\ADNI_meta\*.xml')
ids = [f.split('.')[-2].split('_')[-1] for f in adni_meta]
adni_meta = np.array(adni_meta)
ids = np.array(ids)
nii_data = glob(r'C:\Users\Fabian\stanford\federated_learning_data\ADNI\**\*.nii', recursive=True)
test1 = get_meta_xml(nii_data[0], adni_meta=adni_meta, ids=ids)
print(test1)
test2 = get_meta_xml(nii_data[0], adni_meta=adni_meta, ids=ids, what=['manufacturer', 'model'])
print(test2)
label = get_amyloid_label(rid=test1['rid'], examdate=test1['examdate'], berkeley_data=berkeley_data)
print('nice')
labels = get_labels_from_nifti(nii_data, berkeley_data, adni_meta)
print('done')