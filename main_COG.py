import dataloaders


image_path = '/shares/onfnas01/Research/Bradshaw/COGData/AHOD0831/data_for_nnUNet/nnUNet_raw_data/Task501_COG/imagesTr/'
label_path = '/shares/onfnas01/Research/Bradshaw/COGData/AHOD0831/data_for_nnUNet/nnUNet_raw_data/Task501_COG/labelsTrCopy/'
data_csv_path = '/shares/onfnas01/Research/Bradshaw/COGData/AHOD0831/data_for_nnUNet/nnUNet_raw_data/Task501_COG/data_paths.csv'

df = dataloaders.create_data_list_petct(image_path, label_path, patient_id_template='COG_123456', pet_substring='0001',
                            ct_substring='0002', label_substring='contour',
                            nifti_ext='.nii', save_path=data_csv_path)

