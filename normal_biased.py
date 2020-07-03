# Nicola Dinsdale 2020
# Unlearning with biased datasets dataset
########################################################################################################################
from models.age_predictor import DomainPredictor, Regressor, Encoder
from datasets.numpy_dataset import numpy_dataset_three, numpy_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import numpy as np
from sklearn.utils import shuffle
from utils import Args, EarlyStopping_unlearning
from losses.confusion_loss import confusion_loss
import torch.optim as optim
from train_utils import train_unlearn_distinct, val_unlearn_distinct, val_encoder_domain_unlearn_distinct, train_encoder_domain_unlearn_distinct
import sys

########################################################################################################################
# Create an args class
args = Args()
args.channels_first = True
args.epochs = 300
args.batch_size = 16
args.diff_model_flag = False
args.alpha = 1
args.patience = 150
args.learning_rate = 1e-4

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

# Load in the data
X_biobank = np.load('X_train.npy')
X_oasis = np.load('oasis_X_train.npy')

y_biobank = np.load('y_train.npy').astype(float).reshape(-1)
y_oasis = np.load('oasis_y_train.npy').astype(float).reshape(-1)

# Create the biased datasets
X_biobank = X_biobank[y_biobank < 75]
y_biobank = y_biobank[y_biobank < 75]
X_oasis = X_oasis[y_oasis > 60]
y_oasis = y_oasis[y_oasis > 60]

y_biobank = np.reshape(y_biobank, (-1, 1))
y_oasis = np.reshape(y_oasis, (-1, 1))

print('Biobank shape: ', X_biobank.shape, flush=True)
print('Oasis shape: ', X_oasis.shape, flush=True)

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
    print('CHANNELS FIRST')
    print('Biobank Data shape: ', X_biobank.shape)
    X_oasis = np.transpose(X_oasis, (0, 4, 1, 2, 3))
    print('Oasis Data shape: ', X_oasis.shape)

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

# Now make the overlapping data to do the unlearning with
X_biobank_int = np.load('X_train.npy')
X_oasis_int = np.load('oasis_X_train.npy')

y_biobank_int = np.load('y_train.npy').reshape(-1).astype(float)
y_oasis_int = np.load('oasis_y_train.npy').reshape(-1).astype(float)

indexs = np.where( (y_biobank_int > 60) & (y_biobank_int < 75))
X_biobank_int = X_biobank_int[indexs]
y_biobank_int = y_biobank_int[indexs]
indexs = np.where( (y_oasis_int > 60) & (y_oasis_int < 75))
X_oasis_int = X_oasis_int[indexs]
y_oasis_int= y_oasis_int[indexs]

print('Biobank shape: ', X_biobank_int.shape, flush=True)
print('Oasis shape: ', X_oasis_int.shape, flush=True)

d_biobank_int = np.zeros((len(X_biobank_int), 2))
d_biobank_int[:,0] = 1
d_oasis_int = np.zeros((len(X_oasis_int), 2))
d_oasis_int[:, 1] = 1
d_biobank_int = d_biobank_int.astype(int)
d_oasis_int = d_oasis_int.astype(int)
print(d_biobank_int.shape)
print(d_oasis_int.shape)

if args.channels_first:
    X_biobank_int = np.transpose(X_biobank_int, (0, 4, 1, 2, 3))
    print('CHANNELS FIRST')
    print('Biobank Data shape: ', X_biobank_int.shape)
    X_oasis_int = np.transpose(X_oasis_int, (0, 4, 1, 2, 3))
    print('Oasis Data shape: ', X_oasis_int.shape)

X_biobank_int, d_biobank_int = shuffle(X_biobank_int, d_biobank_int, random_state=0)
X_oasis_int, d_oasis_int = shuffle(X_oasis_int, d_oasis_int, random_state=0)

proportion = int(args.train_val_prop * len(X_biobank_int))
X_btrain_int = X_biobank_int[:proportion, :, :, :, :]
X_bval_int = X_biobank_int[proportion:, :, :, :, :]
d_btrain_int = d_biobank_int[:proportion]
d_bval_int = d_biobank_int[proportion:]

proportion = int(args.train_val_prop * len(X_oasis_int))
X_otrain_int = X_oasis_int[:proportion, :, :, :, :]
X_oval_int = X_oasis_int[proportion:, :, :, :, :]
d_otrain_int = d_oasis_int[:proportion]
d_oval_int = d_oasis_int[proportion:]

print('Data splits')
print(X_btrain_int.shape, d_btrain_int.shape)
print(X_bval_int.shape, d_bval_int.shape)
print(X_otrain_int.shape,  d_otrain_int.shape)
print(X_oval_int.shape, d_oval_int.shape)

b_int_train_dataset = numpy_dataset(X_btrain_int, d_btrain_int)
b_int_val_dataset = numpy_dataset(X_bval_int, d_bval_int)
o_int_train_dataset = numpy_dataset(X_otrain_int, d_otrain_int)
o_int_val_dataset = numpy_dataset(X_oval_int, d_oval_int)

b_int_train_dataloader = DataLoader(b_int_train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
b_int_val_dataloader = DataLoader(b_int_val_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
o_int_train_dataloader = DataLoader(o_int_train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
o_int_val_dataloader = DataLoader(o_int_val_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)

# Load in the model
encoder = Encoder()
regressor = Regressor()
domain_predictor = DomainPredictor(nodes=2)

if cuda:
    encoder = encoder.cuda()
    regressor = regressor.cuda()
    domain_predictor = domain_predictor.cuda()

# Make everything parallelisable
encoder = nn.DataParallel(encoder)
regressor = nn.DataParallel(regressor)
domain_predictor = nn.DataParallel(domain_predictor)

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

criteron = nn.MSELoss()     # Change this to DANN_loss normal
domain_criterion = nn.BCELoss()
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

epoch_reached = args.epoch_reached  # Change this back to 1
loss_store = []

models = [encoder, regressor, domain_predictor]
optimizers = [optimizer, optimizer_conf, optimizer_dm]
train_dataloaders = [b_train_dataloader, o_train_dataloader, b_int_train_dataloader, o_int_train_dataloader]
val_dataloaders = [b_val_dataloader, o_val_dataloader, b_int_val_dataloader, o_int_val_dataloader]
criterions = [criteron, conf_criterion, domain_criterion]

for epoch in range(epoch_reached, args.epochs+1):
    if epoch < args.epoch_stage_1:
        print('Training Main Encoder')
        print('Epoch ', epoch, '/', args.epochs, flush=True)
        optimizers = [optimizer_step1]
        loss, acc, dm_loss, conf_loss = train_encoder_domain_unlearn_distinct(args, models, train_dataloaders, optimizers, criterions, epoch)
        torch.cuda.empty_cache()  # Clear memory cache
        val_loss, val_acc = val_encoder_domain_unlearn_distinct(args, models, val_dataloaders, criterions)
        loss_store.append([loss, val_loss, acc, val_acc, dm_loss, conf_loss])

        # Save the losses each epoch so we can plot them live
        np.save(LOSS_PATH, np.array(loss_store))

        if epoch == args.epoch_stage_1 - 1:
            torch.save(encoder.state_dict(), PRE_TRAIN_ENCODER)
            torch.save(regressor.state_dict(), PRE_TRAIN_REGRESSOR)
            torch.save(domain_predictor.state_dict(), PRE_TRAIN_DOMAIN)
    else:
        optimizer = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()), lr=1e-5)
        optimizer_conf = optim.Adam(list(encoder.parameters()), lr=1e-5)
        optimizer_dm = optim.Adam(list(domain_predictor.parameters()), lr=1e-5)
        optimizers = [optimizer, optimizer_conf, optimizer_dm]

        print('Unlearning')
        print('Epoch ', epoch, '/', args.epochs, flush=True)

        loss, acc, dm_loss, conf_loss = train_unlearn_distinct(args, models, train_dataloaders, optimizers, criterions, epoch)
        torch.cuda.empty_cache()  # Clear memory cache
        val_loss, val_acc = val_unlearn_distinct(args, models, val_dataloaders, criterions)

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

