# Nicola Dinsdale 2020
# Unlearning main with three datasets
########################################################################################################################
from models.age_predictor import DomainPredictor, Regressor, Encoder
from datasets.numpy_dataset import numpy_dataset_three
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import numpy as np
from sklearn.utils import shuffle
from utils import Args, EarlyStopping_unlearning
from losses.confusion_loss import confusion_loss
from losses.DANN_loss import DANN_loss_three_classes
import torch.optim as optim
from train_utils import train_unlearn_threedatasets, val_unlearn_threedatasets, train_encoder_unlearn_threedatasets, val_encoder_unlearn_threedatasets
import sys

########################################################################################################################
# Create an args class
args = Args()
args.channels_first = True
args.epochs = 300
args.batch_size = 16
args.diff_model_flag = False
args.alpha = 1
args.patience = 50
args.learning_rate = 1e-4
args.beta = 10
args.epoch_stage_1 = 100
args.epoch_reached = 1

LOAD_PATH_ENCODER = None
LOAD_PATH_REGRESSOR = None
LOAD_PATH_DOMAIN = None

PRE_TRAIN_ENCODER = 'pretrain_encoder'
PATH_ENCODER = 'encoder_pth'
CHK_PATH_ENCODER = 'encoder_chk_pth'
PRE_TRAIN_REGRESSOR = 'pretrain_regressor'
PATH_REGRESSOR = 'regressor_pth'
CHK_PATH_REGRESSOR = 'regressor_chk_pth'
PRE_TRAIN_DOMAIN = 'pretrain_domain'
PATH_DOMAIN = 'domain_pth'
CHK_PATH_DOMAIN = 'domain_chk_pth'

LOSS_PATH = 'loss_pth'

cuda = torch.cuda.is_available()
if cuda:
    print('Cuda Available', flush=True)

########################################################################################################################
im_size = (128, 128, 32)

# First lets load in the biobank data
X_biobank = np.load('X_train.npy')
X_biobank = np.reshape(X_biobank, (-1, 128, 128, 32, 1))

X_oasis = np.load('oasis_X_train.npy')
X_oasis = np.reshape(X_oasis, (-1, 128, 128, 32, 1))

X_whitehall = np.load('X_train_whitehall1.npy')
X_whitehall = np.reshape(X_whitehall, (-1, 128, 128, 32, 1))

y_biobank = np.load('y_train.npy').reshape(-1, 1).astype(float)
y_oasis = np.load('oasis_y_train.npy').reshape(-1, 1).astype(float)
y_whitehall = (np.load('y_train_whitehall1.npy')).reshape(-1, 1).astype(float)

print('Biobank shape: ', X_biobank.shape, flush=True)
print('Oasis shape: ', X_oasis.shape, flush=True)
print('Whitehall shape:', X_whitehall.shape, flush=True)

d_biobank = np.ones(len(X_biobank)) * 0
d_oasis = np.ones(len(X_biobank)) * 1
d_whitehall = np.ones(len(X_biobank)) * 2

d_biobank = d_biobank.astype(int)
d_oasis = d_oasis.astype(int)
d_whitehall = d_whitehall.astype(int)
print(d_biobank.shape)
print(d_oasis.shape)
print(d_whitehall.shape)

if args.channels_first:
    X_biobank= np.transpose(X_biobank, (0, 4, 1, 2, 3))
    print('CHANNELS FIRST')
    print('Biobank Data shape: ', X_biobank.shape)
    X_oasis = np.transpose(X_oasis, (0, 4, 1, 2, 3))
    print('Oasis Data shape: ', X_oasis.shape)
    X_whitehall = np.transpose(X_whitehall, (0, 4, 1, 2, 3))
    print('Whitehall data shape: ', X_whitehall.shape)

X_biobank, y_biobank, d_biobank = shuffle(X_biobank, y_biobank, d_biobank, random_state=0)
X_oasis, y_oasis, d_oasis = shuffle(X_oasis, y_oasis, d_oasis, random_state=0)
X_whitehall, y_whitehall, d_whitehall = shuffle(X_whitehall, y_whitehall, d_whitehall, random_state=0)

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

proportion = int(args.train_val_prop * len(X_whitehall))
X_wtrain = X_whitehall[:proportion, :, :, :, :]
X_wval = X_whitehall[proportion:, :, :, :, :]
y_wtrain = y_whitehall[:proportion]
y_wval = y_whitehall[proportion:]
d_wtrain = d_whitehall[:proportion]
d_wval = d_whitehall[proportion:]

print('Data splits')
print(X_btrain.shape, y_btrain.shape, d_btrain.shape)
print(X_bval.shape, y_bval.shape, d_bval.shape)
print(X_otrain.shape, y_otrain.shape, d_otrain.shape)
print(X_oval.shape, y_oval.shape, d_oval.shape)
print(X_wtrain.shape, y_wtrain.shape, d_wtrain.shape)
print(X_wval.shape, y_wval.shape, d_wval.shape)

print('Creating datasets and dataloaders')
b_train_dataset = numpy_dataset_three(X_btrain, y_btrain, d_btrain)
b_val_dataset = numpy_dataset_three(X_bval, y_bval, d_bval)
o_train_dataset = numpy_dataset_three(X_otrain, y_otrain, d_otrain)
o_val_dataset = numpy_dataset_three(X_oval, y_oval, d_oval)
w_train_dataset = numpy_dataset_three(X_wtrain, y_wtrain, d_wtrain)
w_val_dataset = numpy_dataset_three(X_wval, y_wval, d_wval)

b_train_dataloader = DataLoader(b_train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
b_val_dataloader = DataLoader(b_val_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
o_train_dataloader = DataLoader(o_train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
o_val_dataloader = DataLoader(o_val_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
w_train_dataloader = DataLoader(w_train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
w_val_dataloader = DataLoader(w_val_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)

# Load the model
encoder = Encoder()
regressor = Regressor()
domain_predictor = DomainPredictor(nodes=3)

if cuda:
    encoder = encoder.cuda()
    regressor = regressor.cuda()
    domain_predictor = domain_predictor.cuda()

if LOAD_PATH_ENCODER:
    print('Loading Weights')
    encoder_dict = encoder.state_dict()
    pretrained_dict = torch.load(LOAD_PATH_ENCODER)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
    print('weights loaded encoder = ', len(pretrained_dict), '/', len(encoder_dict))
    encoder.load_state_dict(torch.load(LOAD_PATH_ENCODER))

if LOAD_PATH_REGRESSOR:
    regressor_dict = regressor.state_dict()
    pretrained_dict = torch.load(LOAD_PATH_REGRESSOR)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in regressor_dict}
    print('weights loaded regressor = ', len(pretrained_dict), '/', len(regressor_dict))
    regressor.load_state_dict(torch.load(LOAD_PATH_REGRESSOR))

if LOAD_PATH_DOMAIN:
    domain_dict = domain_predictor.state_dict()
    pretrained_dict = torch.load(LOAD_PATH_DOMAIN)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in domain_dict}
    print('weights loaded domain predictor = ', len(pretrained_dict), '/', len(domain_dict))
    domain_predictor.load_state_dict(torch.load(LOAD_PATH_DOMAIN))

criteron = nn.MSELoss()
domain_criterion = nn.CrossEntropyLoss()
conf_criterion = confusion_loss()

if cuda:
    criteron = criteron.cuda()
    domain_criterion = domain_criterion.cuda()
    conf_criterion = conf_criterion.cuda()

optimizer_step1 = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()) + list(domain_predictor.parameters()), lr=args.learning_rate)
optimizer = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()), lr=1e-4)
optimizer_conf = optim.Adam(list(encoder.parameters()), lr=1e-4)
optimizer_dm = optim.Adam(list(domain_predictor.parameters()), lr=1e-4)         # Lower learning rate for the unlearning bit

# Initalise the early stopping
early_stopping = EarlyStopping_unlearning(args.patience, verbose=False)

loss_store = []

models = [encoder, regressor, domain_predictor]
optimizers = [optimizer, optimizer_conf, optimizer_dm]
train_dataloaders = [b_train_dataloader, o_train_dataloader, w_train_dataloader]
val_dataloaders = [b_val_dataloader, o_val_dataloader, w_val_dataloader]
criterions = [criteron, conf_criterion, domain_criterion]

for epoch in range(args.epoch_reached, args.epochs+1):
    if epoch < args.epoch_stage_1:
        print('Training Main Encoder')
        print('Epoch ', epoch, '/', args.epochs, flush=True)
        optimizers = [optimizer_step1]
        loss, acc, dm_loss, conf_loss = train_encoder_unlearn_threedatasets(args, models, train_dataloaders, optimizers, criterions, epoch)
        torch.cuda.empty_cache()  # Clear memory cache
        val_loss, val_acc = val_encoder_unlearn_threedatasets(args, models, val_dataloaders, criterions)
        loss_store.append([loss, val_loss, acc, val_acc, dm_loss, conf_loss])

        np.save(LOSS_PATH, np.array(loss_store))

        if epoch == args.epoch_stage_1 - 1:
            torch.save(encoder.state_dict(), PRE_TRAIN_ENCODER)
            torch.save(regressor.state_dict(), PRE_TRAIN_REGRESSOR)
            torch.save(domain_predictor.state_dict(), PRE_TRAIN_DOMAIN)
    else:
        optimizer = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()), lr=1e-6)
        optimizer_conf = optim.Adam(list(encoder.parameters()), lr=1e-6)
        optimizer_dm = optim.Adam(list(domain_predictor.parameters()), lr=1e-6)
        optimizers = [optimizer, optimizer_conf, optimizer_dm]

        print('Unlearning')
        print('Epoch ', epoch, '/', args.epochs, flush=True)

        loss, acc, dm_loss, conf_loss = train_unlearn_threedatasets(args, models, train_dataloaders, optimizers, criterions, epoch)
        torch.cuda.empty_cache()  # Clear memory cache
        val_loss, val_acc = val_unlearn_threedatasets(args, models, val_dataloaders, criterions)

        loss_store.append([loss, val_loss, acc, val_acc, dm_loss, conf_loss])
        np.save(LOSS_PATH, np.array(loss_store))

        # Decide whether the model should stop training or not
        early_stopping(val_loss, models , epoch, optimizer, loss, [CHK_PATH_ENCODER, CHK_PATH_REGRESSOR, CHK_PATH_DOMAIN])
        if early_stopping.early_stop:
            loss_store = np.array(loss_store)
            np.save(LOSS_PATH, loss_store)
            sys.exit('Patience Reached - Early Stopping Activated')

        if epoch == args.epochs:
            print('Finished Training', flush=True)
            print('Saving the model', flush=True)

            # Save the model in such a way that we can continue training later
            torch.save(encoder.state_dict(), PATH_ENCODER)
            torch.save(regressor.state_dict(), PATH_REGRESSOR)
            torch.save(domain_predictor.state_dict(), PATH_DOMAIN)

            loss_store = np.array(loss_store)
            np.save(LOSS_PATH, loss_store)

        torch.cuda.empty_cache()  # Clear memory cache