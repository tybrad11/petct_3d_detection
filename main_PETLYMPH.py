from dataloaders import (create_data_list_petct_when_each_folder_is_one_exam,
                         convert_csv_to_train_val_test_dict_list_with_random_folds,
                         resample_ct_nii_to_match_pet_nii,
                         get_train_val_test_loaders_from_dict_pet_ct_segmentation,
                         get_path_to_Bradshaw_based_on_computer,
                         read_json_into_list_of_dict,
                         check_data_loader_petct_segmentation)
from training import (train_segmentation_model_with_validation)
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai import transforms
from monai.utils import first, set_determinism
import torch
import os

#***********************
#select analysis options
create_data_list_from_folder = 0
separate_into_baseline_and_followup = 0
resample_ct_to_pet = 0
convert_csv_to_dict_list = 0
train_model = 1
#**********************

###### get paths ######
path_to_Bradshaw = get_path_to_Bradshaw_based_on_computer()

image_path = os.path.join(path_to_Bradshaw, 'Lymphoma_UW_Retrospective/Data/uw_analyzed/nifti')
label_path = os.path.join(path_to_Bradshaw, 'Lymphoma_UW_Retrospective/Data/uw_analyzed/nifti')
data_csv_path = os.path.join(path_to_Bradshaw, 'Lymphoma_UW_Retrospective/Analysis/uw_analyzed/analysis1/data_paths.csv')
data_csv_path_baseline = os.path.join(path_to_Bradshaw, 'Lymphoma_UW_Retrospective/Analysis/uw_analyzed/analysis1/data_paths_baseline.csv')
data_csv_path_followup = os.path.join(path_to_Bradshaw, 'Lymphoma_UW_Retrospective/Analysis/uw_analyzed/analysis1/data_paths_followup.csv')
data_csv_path_baseline_resampled = os.path.join(path_to_Bradshaw, 'Lymphoma_UW_Retrospective/Analysis/uw_analyzed/analysis1/data_paths_baseline_resampled.csv')
data_csv_path_followup_resampled = os.path.join(path_to_Bradshaw, 'Lymphoma_UW_Retrospective/Analysis/uw_analyzed/analysis1/data_paths_followup_resampled.csv')
path_json_dict_baseline = [data_csv_path_baseline_resampled[:-4] + '_train.json',
                           data_csv_path_baseline_resampled[:-4] + '_val.json',
                           data_csv_path_baseline_resampled[:-4] + '_test.json']
########################


if create_data_list_from_folder:
    #create list of paths for data
    df = create_data_list_petct_when_each_folder_is_one_exam(image_path, patient_id_template='petlymph_1234', pet_substring='SUV.',
                                ct_substring='_CT_', label_substring='_combined.',
                                nifti_ext='.nii.gz', save_path=data_csv_path)

if separate_into_baseline_and_followup:
    #separate baseline from follow-up -- baseline are even-numbered
    followup_odd_numbered = []
    baseline_even_numbered = []
    for index, row in df.iterrows():
        id = row['id']
        id_num = int(id[-1])
        if (id_num % 2) != 0:
            followup_odd_numbered.append(index)
        else:
            baseline_even_numbered.append(index)
    df_followup = df.iloc[followup_odd_numbered, :]
    df_baseline = df.iloc[baseline_even_numbered, :]
    df_followup.to_csv(data_csv_path_followup)
    df_baseline.to_csv(data_csv_path_baseline)


if resample_ct_to_pet:
    output_csv = resample_ct_nii_to_match_pet_nii(data_csv_path_baseline)
    output_csv_fu = resample_ct_nii_to_match_pet_nii(data_csv_path_followup)



if convert_csv_to_dict_list:
    dict_list_train, dict_list_val, dict_list_test = \
            convert_csv_to_train_val_test_dict_list_with_random_folds(data_csv_path[:-4] + '_baseline.csv', n_random_folds=5)
else:
    dict_list_train = []
    dict_list_val = []
    dict_list_test = []





if train_model:

    #load dictionary list that has been previously saved -- otherwise need to perform convert_csv_to_dict_list (above)
    if len(dict_list_train) == 0:
        dict_list_train = read_json_into_list_of_dict(path_json_dict_baseline[0])
        dict_list_val = read_json_into_list_of_dict(path_json_dict_baseline[1])
        dict_list_test = read_json_into_list_of_dict(path_json_dict_baseline[2])

    train_loader, val_loader, test_loader = \
        get_train_val_test_loaders_from_dict_pet_ct_segmentation(dict_list_train,
                                                                 dict_list_val,
                                                                 dict_list_test,
                                                                 pix_dim=(2.0, 2.0, 3.0),
                                                                 roi_size=(64,64,64),
                                                                 batch_size=1)
    check_data_loader_petct_segmentation(train_loader)

    max_epochs = 100
    val_interval = 5
    # VAL_AMP = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=2,
        out_channels=1,
        dropout_prob=0.2 )
    loss_function = DiceCELoss(smooth_nr=1e-5, smooth_dr=1e-5, squared_pred=True, to_onehot_y=True, sigmoid=True)
    optimizer = torch.optim.RAdam(model.parameters(), 1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    reporting_metric = DiceMetric(include_background=True, reduction="mean_batch")

    post_process_transforms = transforms.Compose(
            [transforms.Activations(sigmoid=True), transforms.AsDiscrete(threshold=0.5)] )

    save_dir = os.path.join(path_to_Bradshaw, 'Lymphoma_UW_Retrospective/Models/uw_lymphoma_classification/analyzed1')

    _ = train_segmentation_model_with_validation(model, device, train_loader, val_loader, max_epochs, val_interval, loss_function,
                                    optimizer, lr_scheduler, reporting_metric, post_process_transforms, save_dir)

