# Tutorials:
# https://github.com/Project-MONAI/MONAIBootcamp2021


import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism

import torch

#reproducible seed
set_determinism(seed=0)

#transforms are used to read and transform data:
# - LoadImage: Load medical specific formats file from provided path
# - Spacing: Resample input image into the specified pixdim
# - Orientation: Change the image’s orientation into the specified axcodes
# - RandGaussianNoise: Perturb image intensities by adding statistical noises
# - NormalizeIntensity: Intensity Normalization based on mean and standard deviation
# - Affine: Transform image based on the affine parameters
# - Rand2DElastic: Random elastic deformation and affine in 2D
# - Rand3DElastic: Random elastic deformation and affine in 3D


###### TRANSFORMS ######

#loading and transforming data is done using Compose(), which is an ordered set of operations, performed in sequence
trans = Compose([LoadImage(image_only=True), AddChannel(), ToTensor()])      #load image without header, add channel dimension, and converts to tensor
image = trans(filename)   #image is then the output of steps, can look at it
plt.imshow(image[0,:,:,10])

#can create custom transform using class with __call__  and inhereting Transform (and maybe Randomizer), or you can
# use a lambda function

#if you only want to operate on some elements of dataset, you can create dictionary tranforms that operate only on specific keys
#these functions have a d at the end: LoadImaged, AddChanneld, Lambdad

#here is an example of a custom transform. Notice how the initialization is passed the key, but the method gets passed the data,
#even though it's not explicitly passed (it's part of the transform chain):
class SquareIt(MapTransform):
    def __init__(self, keys):
        MapTransform.__init__(self, keys)
        print(f"keys to square it: {self.keys}")

    def __call__(self, x):
        key = self.keys[0]
        data = x[key]
        output = {key: data ** 2}
        return output


square_dataset = Dataset(items, transform=SquareIt(keys='data'))

###### DATESETS ######

#similar to monai.data.Dataset, but includes a transform. They also like to work with dictionarys:

#datasets inherit PyTorch Datasets class. Takes both the data and the transofrms
images = [fn["img"] for fn in filenames]

transform = Compose([LoadImage(image_only=True), AddChannel(), ToTensor()])
ds = Dataset(images, transform)
img_tensor = ds[0]
print(img_tensor.shape, img_tensor.get_device())

#to group images and labels, using same tranformations to both, use ArrayDataset:
images = [fn["img"] for fn in filenames]
segs = [fn["seg"] for fn in filenames]

img_transform = Compose([LoadImage(image_only=True), AddChannel(),
                         RandSpatialCrop((128, 128, 128), random_size=False), RandAdditiveNoise(), ToTensor()])
seg_transform = Compose([LoadImage(image_only=True), AddChannel(),
                         RandSpatialCrop((128, 128, 128), random_size=False), ToTensor()])

ds = ArrayDataset(images, img_transform, segs, seg_transform)
im, seg = ds[0]
plt.imshow(np.hstack([im.numpy()[0, 48], seg.numpy()[0, 48]])

#this can be also accomplished using dictionaries:
#Alternatively, Dataset can be used with dictionary-based transforms to construct a result mapping. For training applications beyond simple input/ground-truth pairs like the above this would be more suitable:

trans = Compose([LoadImaged(fn_keys), AddChanneld(fn_keys), RandAdditiveNoised(("img",)),
                 RandSpatialCropd(fn_keys, (128, 128, 128), random_size=False), ToTensord(fn_keys)])

ds = Dataset(filenames, trans)
item = ds[0]
im, seg = item["img"], item["seg"]
plt.imshow(np.hstack([im.numpy()[0, 48], seg.numpy()[0, 48]]))

#regardless of appraoch, these get passed to dataloader
loader = DataLoader(ds, batch_size=5, num_workers=5)
batch = first(loader)
print(list(batch.keys()), batch["img"].shape)

f, ax = plt.subplots(2, 1, figsize=(8, 4))
ax[0].imshow(np.hstack(batch["img"][:, 0, 64]))
ax[1].imshow(np.hstack(batch["seg"][:, 0, 64]))
#note - batches can be dictionarys with keys

###### CACHING #######

#there are different methods for caching the transforms of a dataset, not clear the differencets:
# PersistentDataset
# SmartCacheDataset
# CacheDataset
# CacheDataset provides a mechanism to pre-load all original data and apply non-random transforms into analyzable tensors loaded in memory prior to starting analysis. The CacheDataset requires all tensor representations of data requested to be loaded into memory at once. The subset of random transforms is applied to the cached components before use. This is the highest performance dataset if all data fit in core memory.
# PersistentDataset processes original data sources through the non-random transforms on first use, and stores these intermediate tensor values to an on-disk persistence representation. The intermediate processed tensors are loaded from disk on each use for processing by the random-transforms for each analysis request. The PersistentDataset has a similar memory footprint to the simple Dataset, with performance characteristics close to the CacheDataset at the expense of disk storage. Additionally, the cost of first time processing of data is distributed across each first use.

#PersistenDataset seems the best


###### LAYERS ######

from monai.networks.layers import Conv, Act, split_args, Pool
Conv[Conv.CONV, num_dimensions](in_channels=1, out_channels=4, kernel_size=3)
Act[Act.PRELU](num_parameters=1, init=0.1)

#example network using MONAI


class MyNetwork(torch.nn.Module):

    def __init__(self, dims=3, in_channels=1, out_channels=8, kernel_size=3, pool_kernel=2, act="relu"):
        super(MyNetwork, self).__init__()
        # convolution
        self.conv = Conv[Conv.CONV, dims](in_channels, out_channels, kernel_size=kernel_size)
        # activation
        act_type, act_args = split_args(act)
        self.act = Act[act_type](**act_args)
        # pooling
        self.pool = Pool[Pool.MAX, dims](pool_kernel)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)
        return x

#or use built-in networks:
net = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=1,
    out_channels=1,
    channels=[8, 16, 32, 64],
    strides=[2, 2, 2],
    act=monai.networks.layers.Act.LEAKYRELU
)



##### WORKFLOW ######

monai.data.utils.partition_dataset_classes(data, classes, ratios=None, num_partitions=None, shuffle=False, seed=0, drop_last=False, even_divisible=False)
# data can be indeces, and then just do this;
train_x = [image_files_list[i] for i in train_inds]
train_y = [image_class[i] for i in train_inds]

#these two tranforms will apply a softmax to whatever is fed to it, and the other converts to one-hot encoding:
act = Compose([EnsureType(), Activations(softmax=True)])
to_onehot = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=num_class)])
#these would then be fed to dataloader

#network and options
net = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_class).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), learning_rate)

#training loop

auc_metric = ROCAUCMetric()

for epoch in range(epoch_num):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{epoch_num}")

    epoch_loss = 0
    step = 1

    steps_per_epoch = len(train_ds) // train_loader.batch_size

    # put the network in train mode; this tells the network and its modules to
    # enable training elements such as normalisation and dropout, where applicable
    net.train()
    for batch_data in train_loader:
        # move the data to the GPU
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)

        # prepare the gradients for this step's back propagation
        optimizer.zero_grad()

        # run the network forwards
        outputs = net(inputs)

        # run the loss function on the outputs
        loss = loss_function(outputs, labels)

        # compute the gradients
        loss.backward()

        # tell the optimizer to update the weights according to the gradients
        # and its internal optimisation strategy
        optimizer.step()

        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size + 1}, training_loss: {loss.item():.4f}")
        step += 1

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    # after each epoch, run our metrics to evaluate it, and, if they are an improvement,
    # save the model out

    # switch off training features of the network for this pass
    net.eval()

    # 'with torch.no_grad()' switches off gradient calculation for the scope of its context
    with torch.no_grad():
        # create lists to which we will concatenate the the validation results
        preds = list()
        labels = list()

        # iterate over each batch of images and run them through the network in evaluation mode
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)

            # run the network
            val_pred = net(val_images)

            preds.append(val_pred)
            labels.append(val_labels)

        # concatenate the predicted labels with each other and the actual labels with each other
        y_pred = torch.cat(preds)
        y = torch.cat(labels)

        # we are using the area under the receiver operating characteristic (ROC) curve to determine
        # whether this epoch has improved the best performance of the network so far, in which case
        # we save the network in this state
        y_onehot = [to_onehot(i) for i in decollate_batch(y)]   #this turns a batch into a list you can iterate through, here they apply a one hot transform
        y_pred_act = [act(i) for i in decollate_batch(y_pred)]

        auc_metric(y_pred_act, y_onehot)
        auc_value = auc_metric.aggregate()
        auc_metric.reset()
        metric_values.append(auc_value)

        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)

        if auc_value > best_metric:
            best_metric = auc_value
            best_metric_epoch = epoch + 1
            torch.save(net.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
            print("saved new best metric network")

        print(
            f"current epoch: {epoch + 1} current AUC: {auc_value:.4f} /"
            f" current accuracy: {acc_metric:.4f} best AUC: {best_metric:.4f} /"
            f" at epoch: {best_metric_epoch}"
        )

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")


##### SAVE  #####

torch.save(net.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
#load
net.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
net.eval()



