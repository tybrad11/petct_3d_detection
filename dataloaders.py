import os
import re
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split
import numpy as np
import json
from monai import transforms
from monai import data
import nibabel as nib
from nibabel.processing import resample_from_to
import socket
from monai.utils import first
from matplotlib import pyplot as plt


def get_path_to_Bradshaw_based_on_computer():
    hostname = socket.gethostname()
    if hostname == 'l-mimrtl-gpu2':
        return os.path.join('/shares/onfnas01/Research/Bradshaw')
    elif hostname == 'OON-DGX01':
        return os.path.join('/mnt/DGXUserData/tjb129/Bradshaw')
    elif hostname == 'tyler_docker_dgx':
        return os.path.join('/Data/Data/Bradshaw')

def all_subdirs_search(top_folder):
    subfolders = [f.path for f in os.scandir(top_folder) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(all_subdirs_search(dirname))
    return subfolders
def all_nifti_search(top_folder, nifti_ext = '.nii'):
    len_ext = len(nifti_ext)
    nii_paths = []
    subfolders = [top_folder]
    subfolders.extend(all_subdirs_search(top_folder))
    for dirname in subfolders:
        nii_files = [f.path for f in os.scandir(dirname) if (not f.is_dir() and f.path[-len_ext:] == nifti_ext)]
        nii_paths.extend(nii_files)
    return nii_paths

def read_json_into_list_of_dict(json_filename):
    with open(json_filename, "r") as read_file:
        jdata = json.load(read_file)
    return jdata

def check_data_loader_petct_segmentation(data_loader):
    # load a case, plot 3 slices of pet, ct, and segmentation map. Save as png.
    case = first(data_loader)
    im = case['image']
    lab = case['label']
    z_len = im.shape[-1]
    z_interval = int(z_len/4)
    fig = plt.figure(figsize=(10,7))
    for i in range(3):
        fig.add_subplot(3,3,(i*3)+1)
        plt.imshow(im[0,0,:,:,(i+1)*z_interval])
        fig.add_subplot(3, 3, (i*3)+2)
        plt.imshow(im[0, 1, :, :, (i+1)*z_interval])
        fig.add_subplot(3, 3, (i*3)+3)
        plt.imshow(lab[0, 0, :, :, (i+1)*z_interval])
    fig.savefig('deleteme.png')
    print('check png image:  xdg-open {}'.format(os.path.join(os.getcwd(), 'deleteme.png')))


def create_data_list_petct_when_patientid_is_in_filename(images_dir,
                                  labels_dir,
                                  patient_id_template,  #numbers are wildcards: 'COG_AHOD1001' -> 'COG_AHODXXXX'
                                  pet_substring='pet',
                                  ct_substring='ctac',
                                  label_substring='contour',
                                  nifti_ext = '.nii',
                                  save_path='./data_paths.csv'):
    #search through a folder (or two if labels are in different directory) and find all nifti files that contain a
    #string matching the patient_id_template. Then find all the pet, ct, and label files for that patient, as long as
    #they contain the patient ID and the _substring (see arguments). Saves it as a csv file: row=patient, column = path

    #get all nifti files in the directory
    all_niftis = all_nifti_search(images_dir, nifti_ext)
    if labels_dir != '':
        all_niftis.extend(all_nifti_search(labels_dir, nifti_ext))

    # search through list for anything matching the template
    id_len = len(patient_id_template)
    id_num_len = sum(c.isdigit() for c in patient_id_template)
    id_num_start = re.search(r'[0-9]', patient_id_template).start()
    r = re.compile(".*" + patient_id_template[0:id_num_start] + "[0-9]{" + str(id_num_len) + "}")
    all_niftis_id = list(filter(r.match, all_niftis)) # Read Note below

    #now search for unique patient ids
    unique_ids = []
    for nii_filename in all_niftis_id:
        idx = re.search(patient_id_template[0:id_num_start] + "[0-9]{" + str(id_num_len) + "}", nii_filename).start()
        id = nii_filename[idx: idx+id_len]
        if id not in unique_ids:
            unique_ids.append(id)

    #now get PET, CT, and label filepaths, save as pandas
    dict_list = []
    rpt = re.compile(".*" + pet_substring)
    rct = re.compile(".*" + ct_substring)
    rlb = re.compile(".*" + label_substring)
    #get search results into list
    pt_list = list(filter(rpt.match, all_niftis))
    ct_list = list(filter(rct.match, all_niftis))
    lb_list = list(filter(rlb.match, all_niftis))
    #loop through ID, find all associated pt, ct, and label
    for i, id in enumerate(unique_ids):
        rid = re.compile(".*" + id)
        pt_i = list(filter(rid.match, pt_list))
        ct_i = list(filter(rid.match, ct_list))
        lb_i = list(filter(rid.match, lb_list))
        if len(pt_i) > 0 and len(ct_i) > 0 and len(lb_i) > 0:
            dict_add = {'id': id, 'pet': pt_i[0], 'ct': ct_i[0], 'label': lb_i[0]}
            dict_list.append(dict_add)
        else:
            print('Missing files for {}:\n    pt:{}\n    ct:{}\n    lb:{}'.format(id, pt_i[0], ct_i[0], lb_i[0]))
    df = pd.DataFrame(data=dict_list, columns=['id', 'pet', 'ct', 'label'])
    print('Saving csv to {}'.format(save_path))
    df.to_csv(save_path)
    return df



def create_data_list_petct_when_each_folder_is_one_exam(images_dir,
                                  patient_id_template,  #numbers are wildcards: 'COG_AHOD1001' -> 'COG_AHODXXXX
                                  pet_substring='pet',
                                  ct_substring='ctac',
                                  label_substring='contour',
                                  nifti_ext = '.nii.gz',
                                  save_path='./data_paths.csv'):
    #search through a folder (or two if labels are in different directory) and find all nifti files that contain a
    #string matching the patient_id_template. Then find all the pet, ct, and label files for that patient, as long as
    #they contain the patient ID and the _substring (see arguments). Saves it as a csv file: row=patient, column = path

    folders = os.listdir(images_dir)
    folders.sort()
    dict_list = []

    for patient in folders:
        patient_path  = os.path.join(images_dir, patient)

        #if not a folder
        if not os.path.isdir(patient_path):
            continue

        #get patientID from folder name using patientID template
        id_len = len(patient_id_template)
        id_num_len = sum(c.isdigit() for c in patient_id_template)
        id_num_start = re.search(r'[0-9]', patient_id_template).start()
        r = re.compile(".*" + patient_id_template[0:id_num_start] + "[0-9]{" + str(id_num_len) + "}")
        match = r.search(patient)
        patient_id = patient[match.start(): match.start() + len(patient_id_template)]

        # get all nifti files in the directory
        all_niftis = all_nifti_search(patient_path, nifti_ext)

        #now search for nifti files that match scan strings
        pt_nii = []
        ct_nii = []
        lb_nii = []
        for nii_filepath in all_niftis:
            nii_filename = os.path.split(nii_filepath)[1]
            #now get PET, CT, and label filepaths, save as pandas
            if pet_substring in nii_filename:
                pt_nii.append(nii_filepath)
            elif ct_substring in nii_filename:
                ct_nii.append(nii_filepath)
            elif label_substring in nii_filename:
                lb_nii.append(nii_filepath)
            else:
                continue


        if len(pt_nii) < 1 or len(pt_nii) > 1:
            print('Too many or few PET images for {}'.format(patient_id))
        elif len(ct_nii) < 1 or len(ct_nii) > 1:
            print('Too many or few CT images for {}'.format(patient_id))
        elif len(lb_nii) < 1 or len(lb_nii) > 1:
            print('Too many or few RTSTRUCTs for {}'.format(patient_id))
        else:
            dict_add = {'id': patient_id, 'pet': pt_nii[0], 'ct': ct_nii[0], 'label': lb_nii[0]}
            dict_list.append(dict_add)

    df = pd.DataFrame(data=dict_list, columns=['id', 'pet', 'ct', 'label'])
    print('Saving csv to {}'.format(save_path))
    df.to_csv(save_path)
    return df




def resample_ct_nii_to_match_pet_nii(csv_filename):
    df = pd.read_csv(csv_filename)
    # order alphabetically
    df = df.sort_values('id')
    df_save = df.copy()

    for i, row_i in df.iterrows():
        ct_path_i = row_i['ct']
        pt_path_i = row_i['pet']

        ct_i = nib.load(ct_path_i)
        pt_i = nib.load(pt_path_i)

        #resample
        ct_rs_i = resample_from_to(ct_i, pt_i)

        #saving
        ct_save_path = ct_path_i[:-7] + '_resampled.nii.gz'
        df_save.loc[i, 'ct'] = ct_save_path
        nib.save(ct_rs_i, ct_save_path)

    output_csv = csv_filename[:-4] + '_resampled.csv'
    df_save.to_csv(output_csv)
    return output_csv


def convert_csv_to_train_val_test_dict_list_with_random_folds(csv_path, n_random_folds):
    dict_list_train = []
    dict_list_val = []
    dict_list_test = []

    df = pd.read_csv(csv_path)
    #order alphabetically
    df.sort_values('id')
    n_cases = len(df)

    #now split into folds
    indeces = np.arange(n_cases)
    seed_list = np.arange(n_random_folds)
    # define the folds
    for fold in range(n_random_folds):
        X_temp, X_test = train_test_split(indeces,
                        test_size = 0.3, random_state = seed_list[fold])
        X_train, X_validate  = train_test_split(X_temp,
                        test_size = 10 / 70, random_state = seed_list[fold])

        for i, row_i in df.iterrows():
            if i in X_train:
                dict_list_train.append({'fold': fold, 'image': [row_i['pet'], row_i['ct']], 'label': row_i['label']})
            elif i in X_test:
                dict_list_test.append({'fold': fold, 'image': [row_i['pet'], row_i['ct']], 'label': row_i['label']})
            elif i in X_validate:
                dict_list_val.append({'fold': fold, 'image': [row_i['pet'], row_i['ct']], 'label': row_i['label']})

    output_filename = csv_path[:-4]
    with open(output_filename + '_train.json', 'w') as fout:
        json.dump(dict_list_train, fout)
    with open(output_filename + '_test.json', 'w') as fout:
        json.dump(dict_list_test, fout)
    with open(output_filename + '_val.json', 'w') as fout:
        json.dump(dict_list_val, fout)

    return dict_list_train, dict_list_val, dict_list_test

def get_train_val_test_loaders_from_dict_pet_ct_segmentation(dict_list_train, dict_list_val, dict_list_test, pix_dim, roi_size, batch_size):
    #dict_train has structure: train_files[0] = {'fold':0, 'image': ['/path/pet.nii', '/path/ct.nii'], 'label': '/path/label.nii'}}

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=pix_dim,
                mode=("bilinear", "nearest")),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=roi_size,
                random_size=False,),
            # # transforms.RandCropByLabelClassesd(
            # #         keys=["image", "label"],
            # #         label_key="label",
            # #         spatial_size=[144,144,144],
            # #         ratios=[2,1],
            # #         num_classes=2,
            # # #         num_samples=1),
            # #
            # transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            # transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            # transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            transforms.ToTensord(["image", "label"])
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=pix_dim,
                mode=("bilinear", "nearest")),
            # transforms.RandSpatialCropd(
            #     keys=["image", "label"],
            #     roi_size=[224, 224, 224],
            #     random_size=False, ),
            transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(["image", "label"])

        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=pix_dim,
                mode=("bilinear", "nearest")),
            # transforms.RandSpatialCropd(
            #     keys=["image", "label"],
            #     roi_size=[224, 224, 224],
            #     random_size=False, ),
            transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(["image", "label"])
        ]
    )

    train_ds = data.Dataset(data=dict_list_train, transform=train_transform)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_ds = data.Dataset(data=dict_list_val, transform=val_transform)
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    test_ds = data.Dataset(data=dict_list_test, transform=test_transform)
    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # check_data_loader_petct_segmentation(train_loader)

    return train_loader, val_loader, test_loader


