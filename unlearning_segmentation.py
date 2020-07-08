# Nicola Dinsdale 2020
# Unlearning main for the segmentation model
########################################################################################################################
from models.unet_model import UNet, segmenter, domain_predictor
from datasets.numpy_dataset import numpy_dataset_three
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import numpy as np
from utils import Args, EarlyStopping_unlearning
from losses.confusion_loss import confusion_loss

from losses.dice_loss import dice_loss
from sklearn.utils import shuffle
import torch.optim as optim
from train_utils_segmentation import train_unlearn, val_unlearn, train_encoder_unlearn, val_encoder_unlearn

import sys
########################################################################################################################
# Create an args class
args = Args()
args.channels_first = True
args.epochs = 300
args.batch_size = 3
args.diff_model_flag = False
args.alpha = 50
args.patience = 25
args.epoch_stage_1 = 100
args.epoch_reached = 1
args.beta = 10

cuda = torch.cuda.is_available()

LOAD_PATH_UNET = None
LOAD_PATH_SEGMENTER = None
LOAD_PATH_DOMAIN = None

PRETRAIN_UNET = 'pretrain_unet'
PATH_UNET = 'unet_pth'
CHK_PATH_UNET = 'unet_pth_checkpoint'
PATH_SEGMENTER = 'segmenter_pth'
CHK_PATH_SEGMENTER = 'segmenter_pth_checkpoint'
PRETRAIN_SEGMENTER = 'pretrain_segmenter'
PATH_DOMAIN = 'domain_pth'
CHK_PATH_DOMAIN = 'domain_pth_checkpoint'
PRETRAIN_DOMAIN = 'pretrain_domain'

LOSS_PATH = 'losses'
########################################################################################################################
im_size = (128, 128, 128)
# Load in the data
X_biobank = np.load('X_biobank.npy')        # T1 image
y_biobank = np.load('y_biobank.npy')        # 1 hot labels
y_biobank = np.reshape(y_biobank, (-1, 128, 128, 128, 4))

# Load in the data
X_oasis = np.load('X_oasis.npy')
y_oasis = np.load('y_oasis.npy')

print('Biobank shape: ', X_biobank.shape, flush=True)
print('Oasis shape: ', X_oasis.shape, flush=True)

# Create domain labels
d_biobank = np.zeros((len(X_biobank), 2))
d_biobank[:,0] = 1
d_oasis = np.zeros((len(X_oasis), 2))
d_oasis[:, 1] = 1
d_biobank = d_biobank.astype(int)
d_oasis = d_oasis.astype(int)
print(d_biobank.shape)
print(d_oasis.shape)

if args.channels_first:
    X_biobank= np.transpose(X_biobank, (0, 4, 1, 2, 3))
    y_biobank= np.transpose(y_biobank, (0, 4, 1, 2, 3))
    print('CHANNELS FIRST')
    print('Biobank Data shape: ', X_biobank.shape)
    print('Labels shape: ', y_biobank.shape)
    X_oasis = np.transpose(X_oasis, (0, 4, 1, 2, 3))
    y_oasis = np.transpose(y_oasis, (0, 4, 1, 2, 3))
    print('Oasis Data shape: ', X_oasis.shape)
    print('Labels shape: ', y_oasis.shape)

X_biobank, y_biobank, d_biobank = shuffle(X_biobank, y_biobank, d_biobank, random_state=0)
X_oasis, y_oasis, d_oasis = shuffle(X_oasis, y_oasis, d_oasis, random_state=0)

proportion = int(args.train_val_prop * len(X_biobank))
X_btrain = X_biobank[:proportion, :, :, :, :]
X_bval = X_biobank[proportion:, :, :, :, :]
y_btrain = y_biobank[:proportion]
y_bval = y_biobank[proportion:]
d_btrain = d_biobank[:proportion]
d_bval = d_biobank[proportion:]

proportion = int(args.train_val_prop * len(X_oasis))
X_otrain = X_oasis[:proportion, :, :, :, :]
X_oval = X_oasis[proportion:, :, :, :, :]
y_otrain = y_oasis[:proportion]
y_oval = y_oasis[proportion:]
d_otrain = d_oasis[:proportion]
d_oval = d_oasis[proportion:]

print('Data splits')
print(X_btrain.shape, y_btrain.shape, d_btrain.shape)
print(X_bval.shape, y_bval.shape, d_bval.shape)
print(X_otrain.shape, y_otrain.shape, d_otrain.shape)
print(X_oval.shape, y_oval.shape, d_oval.shape)

print('Creating datasets and dataloaders')
b_train_dataset = numpy_dataset_three(X_btrain, y_btrain, d_btrain)
b_val_dataset = numpy_dataset_three(X_bval, y_bval, d_bval)
o_train_dataset = numpy_dataset_three(X_otrain, y_otrain, d_otrain)
o_val_dataset = numpy_dataset_three(X_oval, y_oval, d_oval)

b_train_dataloader = DataLoader(b_train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
b_val_dataloader = DataLoader(b_val_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
o_train_dataloader = DataLoader(o_train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
o_val_dataloader = DataLoader(o_val_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)

# Load the model
unet = UNet()
segmenter = segmenter()
domain_pred = domain_predictor(2)

if cuda:
    unet = unet.cuda()
    segmenter = segmenter.cuda()
    domain_pred = domain_pred.cuda()

# Make everything parallelisable
unet = nn.DataParallel(unet)
segmenter = nn.DataParallel(segmenter)
domain_pred = nn.DataParallel(domain_pred)

if LOAD_PATH_UNET:
    print('Loading Weights')
    encoder_dict = unet.state_dict()
    pretrained_dict = torch.load(LOAD_PATH_UNET)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
    print('weights loaded encoder = ', len(pretrained_dict), '/', len(encoder_dict))
    unet.load_state_dict(torch.load(LOAD_PATH_UNET))

if LOAD_PATH_SEGMENTER:
    regressor_dict = segmenter.state_dict()
    pretrained_dict = torch.load(LOAD_PATH_SEGMENTER)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in regressor_dict}
    print('weights loaded regressor = ', len(pretrained_dict), '/', len(regressor_dict))
    segmenter.load_state_dict(torch.load(LOAD_PATH_SEGMENTER))

if LOAD_PATH_DOMAIN:
    domain_dict = domain_pred.state_dict()
    pretrained_dict = torch.load(LOAD_PATH_DOMAIN)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in domain_dict}
    print('weights loaded domain predictor = ', len(pretrained_dict), '/', len(domain_dict))
    domain_pred.load_state_dict(torch.load(LOAD_PATH_DOMAIN))

criteron = dice_loss()
criteron.cuda()
domain_criterion = nn.BCELoss()
domain_criterion.cuda()
conf_criterion = confusion_loss()
conf_criterion.cuda()

optimizer_step1 = optim.Adam(list(unet.parameters()) + list(segmenter.parameters()) + list(domain_pred.parameters()), lr=args.learning_rate)
optimizer = optim.Adam(list(unet.parameters()) + list(segmenter.parameters()), lr=1e-4)
optimizer_conf = optim.Adam(list(unet.parameters()), lr=1e-4)
optimizer_dm = optim.Adam(list(domain_pred.parameters()), lr=1e-4)         # Lower learning rate for the unlearning bit

# Initalise the early stopping
early_stopping = EarlyStopping_unlearning(args.patience, verbose=False)

loss_store = []

models = [unet, segmenter, domain_pred]
optimizers = [optimizer, optimizer_conf, optimizer_dm]
train_dataloaders = [b_train_dataloader, o_train_dataloader]
val_dataloaders = [b_val_dataloader, o_val_dataloader]
criterions = [criteron, conf_criterion, domain_criterion]


for epoch in range(args.epoch_reached, args.epochs+1):
    if epoch < args.epoch_stage_1:
        print('Training Main Encoder')
        print('Epoch ', epoch, '/', args.epochs, flush=True)
        optimizers = [optimizer_step1]
        loss, acc, dm_loss, conf_loss = train_encoder_unlearn(args, models, train_dataloaders, optimizers, criterions, epoch)
        torch.cuda.empty_cache()  # Clear memory cache
        val_loss, val_acc = val_encoder_unlearn(args, models, val_dataloaders, criterions)
        loss_store.append([loss, val_loss, acc, val_acc, dm_loss, conf_loss])

        # Save the losses each epoch so we can plot them live
        np.save(LOSS_PATH, np.array(loss_store))

        if epoch == args.epoch_stage_1 - 1:
            torch.save(unet.state_dict(), PRETRAIN_UNET)
            torch.save(segmenter.state_dict(), PRETRAIN_SEGMENTER)
            torch.save(domain_pred.state_dict(), PRETRAIN_DOMAIN)

    else:
        optimizer = optim.Adam(list(unet.parameters()) + list(segmenter.parameters()), lr=1e-5)
        optimizer_conf = optim.Adam(list(unet.parameters()), lr=1e-6)
        optimizer_dm = optim.Adam(list(domain_pred.parameters()), lr=1e-6)
        optimizers = [optimizer, optimizer_conf, optimizer_dm]

        print('Unlearning')
        print('Epoch ', epoch, '/', args.epochs, flush=True)
        optimizers = [optimizer, optimizer_conf, optimizer_dm]

        loss, acc, dm_loss, conf_loss = train_unlearn(args, models, train_dataloaders, optimizers, criterions, epoch)
        torch.cuda.empty_cache()  # Clear memory cache
        val_loss, val_acc = val_unlearn(args, models, val_dataloaders, criterions)

        loss_store.append([loss, val_loss, acc, val_acc, dm_loss, conf_loss])
        np.save(LOSS_PATH, np.array(loss_store))

        # Decide whether the model should stop training or not
        early_stopping(val_loss, models , epoch, optimizer, loss, [CHK_PATH_UNET, CHK_PATH_SEGMENTER, CHK_PATH_DOMAIN])
        if early_stopping.early_stop:
            loss_store = np.array(loss_store)
            np.save(LOSS_PATH, loss_store)
            sys.exit('Patience Reached - Early Stopping Activated')

        if epoch == args.epochs:
            print('Finished Training', flush=True)
            print('Saving the model', flush=True)

            # Save the model in such a way that we can continue training later
            torch.save(unet.state_dict(), PATH_UNET)
            torch.save(segmenter.state_dict(), PATH_SEGMENTER)
            torch.save(domain_pred.state_dict(), PATH_DOMAIN)

            loss_store = np.array(loss_store)
            np.save(LOSS_PATH, loss_store)

        torch.cuda.empty_cache()  # Clear memory cache