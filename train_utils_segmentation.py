# Nicola Dinsdale 2020
# Functions for training and validating the model
########################################################################################################################
# Import dependencies
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
########################################################################################################################

def train_unlearn(args, models, train_loaders, optimizers, criterions, epoch):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [optimizer, optimizer_conf, optimizer_dm] = optimizers
    [b_train_dataloader, o_train_dataloader] = train_loaders
    [criteron, conf_criterion, domain_criterion] = criterions
    regressor_loss = 0
    domain_loss = 0
    conf_loss = 0

    encoder.train()
    regressor.train()
    domain_predictor.train()

    true_domains = []
    pred_domains = []

    batches = 0
    for batch_idx, ((b_data, b_target, b_domain), (o_data, o_target, o_domain)) in enumerate(zip(b_train_dataloader, o_train_dataloader)):
        if len(b_data) == args.batch_size:

            n1 = np.random.randint(1, len(b_data)-1)
            n2 = len(b_data) - n1

            b_data = b_data[:n1]
            b_target = b_target[:n1]
            b_domain = b_domain[:n1]

            o_data = o_data[:n2]
            o_target = o_target[:n2]
            o_domain = o_domain[:n2]

            data = torch.cat((b_data, o_data), 0)
            target = torch.cat((b_target, o_target), 0)
            domain_target = torch.cat((b_domain, o_domain), 0)

            target = target.type(torch.LongTensor)

            if cuda:
                data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

            data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

            if list(data.size())[0] == args.batch_size :
                batches += 1

                # First update the encoder and regressor
                optimizer.zero_grad()
                features = encoder(data)
                output_pred = regressor(features)

                op_0 = output_pred[:n1]
                target_0 = target[:n1]
                loss_0 = criteron(op_0, target_0)

                op_1 = output_pred[n1:]
                target_1 = target[n1:]
                loss_1 = criteron(op_1, target_1)

                loss_total = loss_0 +loss_1
                loss_total.backward(retain_graph=True)
                optimizer.step()

                # Now update just the domain classifier
                optimizer_dm.zero_grad()
                output_dm = domain_predictor(features.detach())

                loss_dm = domain_criterion(output_dm, domain_target)
                loss_dm.backward(retain_graph=False)
                optimizer_dm.step()

                # Now update just the encoder using the domain loss
                optimizer_conf.zero_grad()
                output_dm_conf = domain_predictor(features)
                loss_conf = args.beta * conf_criterion(output_dm_conf, domain_target)
                loss_conf.backward(retain_graph=False)
                optimizer_conf.step()

                regressor_loss += loss_total
                domain_loss += loss_dm
                conf_loss += loss_conf

                output_dm_conf = np.argmax(output_dm_conf.detach().cpu().numpy(), axis=1)
                domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
                true_domains.append(domain_target)
                pred_domains.append(output_dm_conf)

                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx+1) * len(data), len(b_train_dataloader.dataset),
                               100. * (batch_idx+1) / len(b_train_dataloader), loss_total.item()), flush=True)
                    print('\t \t Confusion loss = ', loss_conf.item())
                    print('\t \t Domain Loss = ', loss_dm.item(), flush=True)
                del target
                del loss_total
                del features

            torch.cuda.empty_cache()  # Clear memory cache

    av_loss = regressor_loss / batches

    av_conf = conf_loss / batches

    av_dom = domain_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss,  flush=True))
    print('\nTraining set: Average Conf loss: {:.4f}'.format(av_conf,  flush=True))
    print('\nTraining set: Average Dom loss: {:.4f}'.format(av_dom,  flush=True))

    print('\nTraining set: Average Acc: {:.4f}\n'.format(acc,  flush=True))

    return av_loss, acc, av_dom, av_conf

def train_unlearn_multi(args, models, train_loaders, optimizers, criterions, epoch):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [optimizer, optimizer_conf, optimizer_dm] = optimizers
    [b_train_dataloader, o_train_dataloader] = train_loaders
    [criteron, conf_criterion, domain_criterion] = criterions
    regressor_loss = 0
    domain_loss = 0
    conf_loss = 0

    encoder.train()
    regressor.train()
    domain_predictor.train()

    true_domains = []
    pred_domains = []

    batches = 0
    for batch_idx, ((b_data, b_target, b_domain), (o_data, o_target, o_domain)) in enumerate(zip(b_train_dataloader, o_train_dataloader)):
        if len(b_data) == args.batch_size:

            n1 = np.random.randint(1, len(b_data)-1)
            n2 = len(b_data) - n1

            b_data = b_data[:n1]
            b_target = b_target[:n1]
            b_domain = b_domain[:n1]

            o_data = o_data[:n2]
            o_target = o_target[:n2]
            o_domain = o_domain[:n2]

            data = torch.cat((b_data, o_data), 0)
            target = torch.cat((b_target, o_target), 0)
            domain_target = torch.cat((b_domain, o_domain), 0)

            target = target.type(torch.LongTensor)

            if cuda:
                data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

            data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

            if list(data.size())[0] == args.batch_size :
                batches += 1

                # First update the encoder and regressor
                optimizer.zero_grad()
                [features, bottleneck] = encoder(data)
                output_pred = regressor(features)

                op_0 = output_pred[:n1]
                target_0 = target[:n1]
                loss_0 = criteron(op_0, target_0)

                op_1 = output_pred[n1:]
                target_1 = target[n1:]
                loss_1 = criteron(op_1, target_1)

                loss_total = loss_0 + loss_1
                loss_total.backward(retain_graph=True)         # change this back to true
                optimizer.step()

                # Now update just the domain classifier
                optimizer_dm.zero_grad()
                output_dm = domain_predictor([features.detach(), bottleneck.detach()])

                loss_dm = domain_criterion(output_dm, domain_target)
                loss_dm.backward(retain_graph=False)
                optimizer_dm.step()

                # Now update just the encoder using the domain loss
                optimizer_conf.zero_grad()
                output_dm_conf = domain_predictor([features, bottleneck])
                loss_conf = args.beta * conf_criterion(output_dm_conf, domain_target)        # Get rid of the weight for not unsupervised
                loss_conf.backward(retain_graph=False)
                optimizer_conf.step()

                regressor_loss += loss_total
                domain_loss += loss_dm
                conf_loss += loss_conf

                output_dm_conf = np.argmax(output_dm_conf.detach().cpu().numpy(), axis=1)
                domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
                true_domains.append(domain_target)
                pred_domains.append(output_dm_conf)

                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx+1) * len(data), len(b_train_dataloader.dataset),
                               100. * (batch_idx+1) / len(b_train_dataloader), loss_total.item()), flush=True)
                    print('\t \t Confusion loss = ', loss_conf.item())
                    print('\t \t Domain Loss = ', loss_dm.item(), flush=True)
                del target
                del loss_total
                del features

            torch.cuda.empty_cache()  # Clear memory cache

    av_loss = regressor_loss / batches

    av_conf = conf_loss / batches

    av_dom = domain_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss,  flush=True))
    print('\nTraining set: Average Conf loss: {:.4f}'.format(av_conf,  flush=True))
    print('\nTraining set: Average Dom loss: {:.4f}'.format(av_dom,  flush=True))

    print('\nTraining set: Average Acc: {:.4f}\n'.format(acc,  flush=True))

    return av_loss, acc, av_dom, av_conf

def val_unlearn(args, models, val_loaders, criterions):

    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [b_val_dataloader, o_val_dataloader] = val_loaders
    [criteron, _, _] = criterions

    encoder.eval()
    regressor.eval()
    domain_predictor.eval()

    val_loss = 0

    true_domains = []
    pred_domains = []

    batches = 0
    with torch.no_grad():
        for batch_idx, ((b_data, b_target, b_domain), (o_data, o_target, o_domain)) in enumerate(zip(b_val_dataloader, o_val_dataloader)):
            if len(b_data) == args.batch_size:

                n1 = np.random.randint(1, len(b_data) - 1)
                n2 = len(b_data) - n1

                b_data = b_data[:n1]
                b_target = b_target[:n1]
                b_domain = b_domain[:n1]

                o_data = o_data[:n2]
                o_target = o_target[:n2]
                o_domain = o_domain[:n2]

                data = torch.cat((b_data, o_data), 0)
                target = torch.cat((b_target, o_target), 0)
                domain_target = torch.cat((b_domain, o_domain), 0)
                target = target.type(torch.LongTensor)

                if cuda:
                    data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

                data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

                if list(data.size())[0] == args.batch_size:
                    batches += 1

                    features = encoder(data)
                    output_pred = regressor(features)

                    op_0 = output_pred[:n1]
                    target_0 = target[:n1]
                    loss_0 = criteron(op_0, target_0)

                    op_1 = output_pred[n1:]
                    target_1 = target[n1:]
                    loss_1 = criteron(op_1, target_1)

                    loss_total = loss_0 + loss_1
                    val_loss += loss_total

                    domains = domain_predictor.forward(features)
                    domains = np.argmax(domains.detach().cpu().numpy(), axis=1)
                    domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
                    true_domains.append(domain_target)
                    pred_domains.append(domains)

            torch.cuda.empty_cache()

    val_loss = val_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss,  flush=True))
    print('Validation set: Average Acc: {:.4f}\n'.format(acc,  flush=True))

    return val_loss, acc

def val_unlearn_multi(args, models, val_loaders, criterions):

    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [b_val_dataloader, o_val_dataloader] = val_loaders
    [criteron, _, _] = criterions

    encoder.eval()
    regressor.eval()
    domain_predictor.eval()

    val_loss = 0

    true_domains = []
    pred_domains = []

    batches = 0
    with torch.no_grad():
        for batch_idx, ((b_data, b_target, b_domain), (o_data, o_target, o_domain)) in enumerate(zip(b_val_dataloader, o_val_dataloader)):
            if len(b_data) == args.batch_size:
                n1 = np.random.randint(1, len(b_data) - 1)
                n2 = len(b_data) - n1

                b_data = b_data[:n1]
                b_target = b_target[:n1]
                b_domain = b_domain[:n1]

                o_data = o_data[:n2]
                o_target = o_target[:n2]
                o_domain = o_domain[:n2]

                data = torch.cat((b_data, o_data), 0)
                target = torch.cat((b_target, o_target), 0)
                domain_target = torch.cat((b_domain, o_domain), 0)
                target = target.type(torch.LongTensor)

                if cuda:
                    data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

                data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

                if list(data.size())[0] == args.batch_size:
                    batches += 1

                    [features, bottleneck] = encoder(data)
                    output_pred = regressor(features)

                    op_0 = output_pred[:n1]
                    target_0 = target[:n1]
                    loss_0 = criteron(op_0, target_0)

                    op_1 = output_pred[n1:]
                    target_1 = target[n1:]
                    loss_1 = criteron(op_1, target_1)

                    loss_total = loss_0 + loss_1
                    val_loss += loss_total

                    domains = domain_predictor.forward([features, bottleneck])
                    domains = np.argmax(domains.detach().cpu().numpy(), axis=1)
                    domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
                    true_domains.append(domain_target)
                    pred_domains.append(domains)

            torch.cuda.empty_cache()  # Clear memory cache

    val_loss = val_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss,  flush=True))
    print('Validation set: Average Acc: {:.4f}\n'.format(acc,  flush=True))

    return val_loss, acc

def train_encoder_unlearn(args, models, train_loaders, optimizers, criterions, epoch):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [optimizer] = optimizers
    [b_train_dataloader, o_train_dataloader] = train_loaders
    [criteron, _, domain_criterion] = criterions
    regressor_loss = 0
    domain_loss = 0

    encoder.train()
    regressor.train()
    domain_predictor.train()

    true_domains = []
    pred_domains = []

    batches = 0
    for batch_idx, ((b_data, b_target, b_domain), (o_data, o_target, o_domain)) in enumerate(zip(b_train_dataloader, o_train_dataloader)):
        if len(b_data) == args.batch_size:
            n1 = np.random.randint(1, len(b_data)-1)
            n2 = len(b_data) - n1

            b_data = b_data[:n1]
            b_target = b_target[:n1]
            b_domain = b_domain[:n1]

            o_data = o_data[:n2]
            o_target = o_target[:n2]
            o_domain = o_domain[:n2]

            data = torch.cat((b_data, o_data), 0)
            target = torch.cat((b_target, o_target), 0)
            domain_target = torch.cat((b_domain, o_domain), 0)
            target = target.type(torch.LongTensor)

            if cuda:
                data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

            data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

            if list(data.size())[0] == args.batch_size :
                batches += 1

                # First update the encoder and regressor
                optimizer.zero_grad()
                features = encoder(data)
                output_pred = regressor(features)

                op_0 = output_pred[:n1]
                target_0 = target[:n1]
                loss_0= criteron(op_0, target_0)

                op_1 = output_pred[n1:]
                target_1 = target[n1:]
                loss_1 = criteron(op_1, target_1)

                r_loss = loss_0 + loss_1
                domain_pred = domain_predictor(features)

                d_loss = domain_criterion(domain_pred, domain_target)
                loss = r_loss + d_loss
                loss.backward()
                optimizer.step()

                regressor_loss += r_loss
                domain_loss += d_loss

                domains = np.argmax(domain_pred.detach().cpu().numpy(), axis=1)
                domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
                true_domains.append(domain_target)
                pred_domains.append(domains)

                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Regressor Loss: {:.6f}'.format(
                        epoch, (batch_idx+1) * len(data), len(b_train_dataloader.dataset),
                               100. * (batch_idx+1) / len(b_train_dataloader), r_loss.item()), flush=True)
                    print('Regressor Loss: {:.4f}'.format(r_loss, flush=True))
                    print('Domain Loss: {:.4f}'.format(d_loss, flush=True))

                del target
                del r_loss
                del d_loss
                del features

    av_loss = regressor_loss / batches

    av_dom_loss = domain_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss,  flush=True))
    print('Training set: Average Domain loss: {:.4f}'.format(av_dom_loss,  flush=True))
    print('Training set: Average Acc: {:.4f}'.format(acc,  flush=True))

    return av_loss, acc, av_dom_loss, np.NaN

def train_encoder_unlearn_multi(args, models, train_loaders, optimizers, criterions, epoch):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [optimizer] = optimizers
    [b_train_dataloader, o_train_dataloader] = train_loaders
    [criteron, _, domain_criterion] = criterions
    regressor_loss = 0
    domain_loss = 0

    encoder.train()
    regressor.train()
    domain_predictor.train()

    true_domains = []
    pred_domains = []

    batches = 0
    for batch_idx, ((b_data, b_target, b_domain), (o_data, o_target, o_domain)) in enumerate(zip(b_train_dataloader, o_train_dataloader)):
        if len(b_data) == args.batch_size:
            n1 = np.random.randint(1, len(b_data)-1)
            n2 = len(b_data) - n1

            b_data = b_data[:n1]
            b_target = b_target[:n1]
            b_domain = b_domain[:n1]

            o_data = o_data[:n2]
            o_target = o_target[:n2]
            o_domain = o_domain[:n2]

            data = torch.cat((b_data, o_data), 0)
            target = torch.cat((b_target, o_target), 0)
            domain_target = torch.cat((b_domain, o_domain), 0)
            target = target.type(torch.LongTensor)

            if cuda:
                data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

            data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

            if list(data.size())[0] == args.batch_size :
                batches += 1

                # First update the encoder and regressor
                optimizer.zero_grad()
                [features, bottleneck] = encoder(data)
                output_pred = regressor(features)

                op_0 = output_pred[:n1]
                target_0 = target[:n1]
                loss_0= criteron(op_0, target_0)

                op_1 = output_pred[n1:]
                target_1 = target[n1:]
                loss_1 = criteron(op_1, target_1)

                r_loss = loss_0 + loss_1
                domain_pred = domain_predictor([features, bottleneck])

                d_loss = domain_criterion(domain_pred, domain_target)
                loss = r_loss + args.beta*d_loss
                loss.backward()
                optimizer.step()

                regressor_loss += r_loss
                domain_loss += d_loss

                domains = np.argmax(domain_pred.detach().cpu().numpy(), axis=1)
                domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
                true_domains.append(domain_target)
                pred_domains.append(domains)

                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Regressor Loss: {:.6f}'.format(
                        epoch, (batch_idx+1) * len(data), len(b_train_dataloader.dataset),
                               100. * (batch_idx+1) / len(b_train_dataloader), r_loss.item()), flush=True)
                    print('Regressor Loss: {:.4f}'.format(r_loss, flush=True))
                    print('Domain Loss: {:.4f}'.format(d_loss, flush=True))

                del target
                del r_loss
                del d_loss
                del features

    av_loss = regressor_loss / batches

    av_dom_loss = domain_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    del av_loss
    del acc
    del av_dom_loss

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss,  flush=True))
    print('Training set: Average Domain loss: {:.4f}'.format(av_dom_loss,  flush=True))
    print('Training set: Average Acc: {:.4f}'.format(acc,  flush=True))

    return av_loss, acc, av_dom_loss, np.NaN

def val_encoder_unlearn(args, models, val_loaders, criterions):

    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [b_val_dataloader, o_val_dataloader] = val_loaders
    [criteron, _, domain_criterion] = criterions

    encoder.eval()
    regressor.eval()
    domain_predictor.eval()

    regressor_loss = 0
    domain_loss = 0

    true_domains = []
    pred_domains = []

    batches = 0
    with torch.no_grad():
        for batch_idx, ((b_data, b_target, b_domain), (o_data, o_target, o_domain)) in enumerate(zip(b_val_dataloader, o_val_dataloader)):
            if len(b_data) == args.batch_size:

                n1 = np.random.randint(1, len(b_data) - 1)
                n2 = len(b_data) - n1

                b_data = b_data[:n1]
                b_target = b_target[:n1]
                b_domain = b_domain[:n1]

                o_data = o_data[:n2]
                o_target = o_target[:n2]
                o_domain = o_domain[:n2]

                data = torch.cat((b_data, o_data), 0)
                target = torch.cat((b_target, o_target), 0)
                domain_target = torch.cat((b_domain, o_domain), 0)
                target = target.type(torch.LongTensor)

                if cuda:
                    data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

                data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

                if list(data.size())[0] == args.batch_size:
                    batches += 1
                    features = encoder(data)
                    output_pred = regressor(features)

                    op_0 = output_pred[:n1]
                    target_0 = target[:n1]
                    loss_0 = criteron(op_0, target_0)

                    op_1 = output_pred[n1:]
                    target_1 = target[n1:]
                    loss_1 = criteron(op_1, target_1)

                    r_loss = loss_0 + loss_1

                    domain_pred = domain_predictor(features)

                    d_loss = domain_criterion(domain_pred, domain_target)

                    domains = np.argmax(domain_pred.detach().cpu().numpy(), axis=1)
                    domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
                    true_domains.append(domain_target)
                    pred_domains.append(domains)

                    regressor_loss += r_loss
                    domain_loss += d_loss

    val_loss = regressor_loss / batches

    dom_loss = domain_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss,  flush=True))
    print('Validation set: Average Domain loss: {:.4f}\n'.format(dom_loss,  flush=True))
    print(' Validation set: Average Acc: {:.4f}'.format(acc,  flush=True))

    return val_loss, acc

def val_encoder_unlearn_multi(args, models, val_loaders, criterions):

    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [b_val_dataloader, o_val_dataloader] = val_loaders
    [criteron, _, domain_criterion] = criterions

    encoder.eval()
    regressor.eval()
    domain_predictor.eval()

    regressor_loss = 0
    domain_loss = 0

    true_domains = []
    pred_domains = []

    batches = 0
    with torch.no_grad():
        for batch_idx, ((b_data, b_target, b_domain), (o_data, o_target, o_domain)) in enumerate(zip(b_val_dataloader, o_val_dataloader)):
            if len(b_data) == args.batch_size:

                n1 = np.random.randint(1, len(b_data) - 1)
                n2 = len(b_data) - n1

                b_data = b_data[:n1]
                b_target = b_target[:n1]
                b_domain = b_domain[:n1]

                o_data = o_data[:n2]
                o_target = o_target[:n2]
                o_domain = o_domain[:n2]

                data = torch.cat((b_data, o_data), 0)
                target = torch.cat((b_target, o_target), 0)
                domain_target = torch.cat((b_domain, o_domain), 0)
                target = target.type(torch.LongTensor)

                if cuda:
                    data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

                data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

                if list(data.size())[0] == args.batch_size:
                    batches += 1
                    [features, bottleneck] = encoder(data)
                    output_pred = regressor(features)

                    op_0 = output_pred[:n1]
                    target_0 = target[:n1]
                    loss_0 = criteron(op_0, target_0)

                    op_1 = output_pred[n1:]
                    target_1 = target[n1:]
                    loss_1 = criteron(op_1, target_1)

                    r_loss = loss_0 + loss_1

                    domain_pred = domain_predictor([features, bottleneck])

                    d_loss = domain_criterion(domain_pred, domain_target)

                    domains = np.argmax(domain_pred.detach().cpu().numpy(), axis=1)
                    domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
                    true_domains.append(domain_target)
                    pred_domains.append(domains)

                    regressor_loss += r_loss
                    domain_loss += d_loss

    val_loss = regressor_loss / batches

    dom_loss = domain_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss,  flush=True))
    print('Validation set: Average Domain loss: {:.4f}\n'.format(dom_loss,  flush=True))
    print(' Validation set: Average Acc: {:.4f}'.format(acc,  flush=True))

    return val_loss, acc

def train_unlearn_semi(args, models, train_loaders, optimizers, criterions, epoch):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [optimizer, optimizer_conf, optimizer_dm] = optimizers
    [b_train_dataloader, o_train_dataloader, b_train_int_dataloader, o_train_int_dataloader] = train_loaders
    [criteron, conf_criterion, domain_criterion] = criterions

    regressor_loss = 0
    domain_loss = 0
    conf_loss = 0

    encoder.train()
    regressor.train()
    domain_predictor.train()

    true_domains = []
    pred_domains = []

    batches = 0
    for batch_idx, ((b_data, b_target, b_domain), (o_data, o_target, o_domain), (b_int_data, b_int_domain), (o_int_data, o_int_domain)) in enumerate(zip(b_train_dataloader, o_train_dataloader, b_train_int_dataloader, o_train_int_dataloader)):
        n1 = np.random.randint(1, len(b_data)-1)
        n2 = len(b_data) - n1

        b_data = b_data[:n1]
        b_target = b_target[:n1]
        b_domain = b_domain[:n1]

        o_data = o_data[:n2]
        o_target = o_target[:n2]
        o_domain = o_domain[:n2]

        b_int_data = b_int_data[:n1]
        b_int_domain = b_int_domain[:n1]
        o_int_data = o_int_data[:n2]
        o_int_domain = o_int_domain[:n2]

        data = torch.cat((b_data, o_data), 0)
        target = torch.cat((b_target, o_target), 0)
        domain_target = torch.cat((b_domain, o_domain), 0)

        int_data = torch.cat((b_int_data, o_int_data), 0)
        int_domain = torch.cat((b_int_domain, o_int_domain), 0)
        target = target.type(torch.LongTensor)

        if cuda:
            data, target, domain_target, int_data, int_domain = data.cuda(), target.cuda(), domain_target.cuda(), int_data.cuda(), int_domain.cuda()

        data, target, domain_target, int_data, int_domain = Variable(data), Variable(target), Variable(domain_target), Variable(int_data), Variable(int_domain)

        if list(data.size())[0] == args.batch_size :
            if list(int_domain.size())[0] == args.batch_size :

                batches += 1

                # First update the encoder and regressor
                optimizer.zero_grad()
                features = encoder(data)
                output_pred = regressor(features)

                op_0 = output_pred[:n1]
                target_0 = target[:n1]
                loss_0 = criteron(op_0, target_0)

                op_1 = output_pred[n1:]
                target_1 = target[n1:]
                loss_1 = criteron(op_1, target_1)

                loss_total = loss_0 + loss_1
                loss_total.backward()
                optimizer.step()

                # Now update just the domain classifier on the intersection data only
                optimizer_dm.zero_grad()
                new_features = encoder(int_data)
                output_dm = domain_predictor(new_features.detach())
                loss_dm = domain_criterion(output_dm, int_domain)
                loss_dm.backward()
                optimizer_dm.step()

                # Now update just the encoder using the domain loss
                optimizer_conf.zero_grad()
                output_dm_conf = domain_predictor(new_features)
                loss_conf = args.beta * conf_criterion(output_dm_conf, int_domain)
                loss_conf.backward(retain_graph=False)
                optimizer_conf.step()

                regressor_loss += loss_total
                domain_loss += loss_dm
                conf_loss += loss_conf

                output_dm_conf = np.argmax(output_dm_conf.detach().cpu().numpy(), axis=1)
                domain_target = np.argmax(int_domain.detach().cpu().numpy(), axis=1)
                true_domains.append(np.array(domain_target))
                pred_domains.append(np.array(output_dm_conf))

                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx+1) * len(data), len(b_train_dataloader.dataset),
                               100. * (batch_idx+1) / len(b_train_dataloader), loss_total.item()), flush=True)
                    print('\t \t Confusion loss = ', loss_conf.item())
                    print('\t \t Domain Loss = ', loss_dm.item(), flush=True)
                del target
                del loss_total
                del features

    av_loss = regressor_loss / batches

    av_conf = loss_conf / batches

    av_dom = loss_dm / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)

    acc = accuracy_score(true_domains, pred_domains)

    print('Training set: Average loss: {:.4f}'.format(av_loss,  flush=True))
    print('Training set: Average Conf loss: {:.4f}'.format(av_conf,  flush=True))
    print('Training set: Average Dom loss: {:.4f}'.format(av_dom,  flush=True))

    print('Training set: Average Acc: {:.4f}\n'.format(acc,  flush=True))

    return av_loss, acc, av_dom, av_conf

def val_unlearn_semi(args, models, val_loaders, criterions):

    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [b_val_dataloader, o_val_dataloader, b_int_val_dataloader, o_int_val_dataloader] = val_loaders
    [criteron, _, _] = criterions

    encoder.eval()
    regressor.eval()
    domain_predictor.eval()

    val_loss = 0

    true_domains = []
    pred_domains = []

    batches = 0
    with torch.no_grad():
        for batch_idx, ((b_data, b_target, b_domain), (o_data, o_target, o_domain), (b_int_data, b_int_domain), (o_int_data, o_int_domain)) in enumerate(zip(b_val_dataloader, o_val_dataloader, b_int_val_dataloader, o_int_val_dataloader)):
            n1 = np.random.randint(1, len(b_data) - 1)
            n2 = len(b_data) - n1

            b_data = b_data[:n1]
            b_target = b_target[:n1]
            b_domain = b_domain[:n1]

            o_data = o_data[:n2]
            o_target = o_target[:n2]
            o_domain = o_domain[:n2]

            b_int_data = b_int_data[:n1]
            b_int_domain = b_int_domain[:n1]
            o_int_data = o_int_data[:n2]
            o_int_domain = o_int_domain[:n2]

            data = torch.cat((b_data, o_data), 0)
            target = torch.cat((b_target, o_target), 0)
            domain_target = torch.cat((b_domain, o_domain), 0)

            int_data = torch.cat((b_int_data, o_int_data), 0)
            int_domain = torch.cat((b_int_domain, o_int_domain), 0)
            target = target.type(torch.LongTensor)

            if cuda:
                data, target, domain_target, int_data, int_domain = data.cuda(), target.cuda(), domain_target.cuda(), int_data.cuda(), int_domain.cuda()

            data, target, domain_target, int_data, int_domain = Variable(data), Variable(target), Variable(domain_target), Variable(int_data), Variable(int_domain)

            if list(data.size())[0] == args.batch_size:
                if list(int_data.size())[0] == args.batch_size:
                    batches += 1
                    features = encoder(data)
                    output_pred = regressor(features)

                    op_0 = output_pred[:n1]
                    target_0 = target[:n1]
                    loss_0 = criteron(op_0, target_0)

                    op_1 = output_pred[n1:]
                    target_1 = target[n1:]
                    loss_1 = criteron(op_1, target_1)

                    loss_total = loss_0 + loss_1
                    val_loss += loss_total

                    new_features = encoder(int_data)
                    domains = domain_predictor.forward(new_features)
                    domains = np.argmax(domains.detach().cpu().numpy(), axis=1)
                    domain_target = np.argmax(int_domain.detach().cpu().numpy(), axis=1)
                    true_domains.append(domain_target)
                    pred_domains.append(domains)

    val_loss = val_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)

    acc = accuracy_score(true_domains, pred_domains)

    print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss,  flush=True))
    print('Validation set: Average Acc: {:.4f}\n'.format(acc,  flush=True))

    return val_loss, acc

def train_encoder_domain_unlearn_semi(args, models, train_loaders, optimizers, criterions, epoch):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [optimizer] = optimizers
    [b_train_dataloader, o_train_dataloader, _, _] = train_loaders
    [criteron, _, domain_criterion] = criterions
    regressor_loss = 0
    domain_loss = 0

    encoder.train()
    regressor.train()
    domain_predictor.train()

    true_domains = []
    pred_domains = []

    batches = 0
    for batch_idx, ((b_data, b_target, b_domain), (o_data, o_target, o_domain)) in enumerate(zip(b_train_dataloader, o_train_dataloader)):
        n1 = np.random.randint(1, len(b_data)-1)
        n2 = len(b_data) - n1

        b_data = b_data[:n1]
        b_target = b_target[:n1]
        b_domain = b_domain[:n1]

        o_data = o_data[:n2]
        o_target = o_target[:n2]
        o_domain = o_domain[:n2]

        data = torch.cat((b_data, o_data), 0)
        target = torch.cat((b_target, o_target), 0)
        domain_target = torch.cat((b_domain, o_domain), 0)
        target = target.type(torch.LongTensor)

        if cuda:
            data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

        data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)


        if list(data.size())[0] == args.batch_size :

            batches += 1

            # First update the encoder and regressor for now dont improve the domain stuff, just the feature predictions
            optimizer.zero_grad()
            features = encoder(data)
            output_pred = regressor(features)

            op_0 = output_pred[:n1]
            target_0 = target[:n1]
            loss_0 = criteron(op_0, target_0)

            op_1 = output_pred[n1:]
            target_1 = target[n1:]
            loss_1 = criteron(op_1, target_1)

            loss = loss_0 + loss_1
            regressor_loss += loss

            output_dm = domain_predictor(features.detach())
            loss_dm = domain_criterion(output_dm, domain_target)

            loss = loss + args.alpha * loss_dm
            loss.backward()
            optimizer.step()

            domain_loss += loss_dm

            output_dm_conf = np.argmax(output_dm.detach().cpu().numpy(), axis=1)
            domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
            true_domains.append(np.array(domain_target))
            pred_domains.append(np.array(output_dm_conf))


            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Regressor Loss: {:.6f}'.format(
                    epoch, (batch_idx+1) * len(data), len(b_train_dataloader.dataset),
                           100. * (batch_idx+1) / len(b_train_dataloader), loss.item()), flush=True)

            del target
            del features
            del loss

    av_loss = regressor_loss / batches

    av_dom = loss_dm / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)

    acc = accuracy_score(true_domains, pred_domains)

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss,  flush=True))
    print('Training set: Average Dom loss: {:.4f}'.format(av_dom,  flush=True))
    print('Training set: Average Acc: {:.4f}\n'.format(acc,  flush=True))

    return av_loss, acc, av_dom, np.NaN

def val_encoder_domain_unlearn_semi(args, models, val_loaders, criterions):

    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [b_val_dataloader, o_val_dataloader, _, _] = val_loaders
    [criteron, _, _] = criterions

    encoder.eval()
    regressor.eval()
    domain_predictor.eval()

    val_loss = 0

    true_domains = []
    pred_domains = []

    batches = 0
    with torch.no_grad():
        for batch_idx, ((b_data, b_target, b_domain), (o_data, o_target, o_domain)) in enumerate(zip(b_val_dataloader, o_val_dataloader)):
            n1 = np.random.randint(1, len(b_data) - 1)
            n2 = len(b_data) - n1

            b_data = b_data[:n1]
            b_target = b_target[:n1]
            b_domain = b_domain[:n1]

            o_data = o_data[:n2]
            o_target = o_target[:n2]
            o_domain = o_domain[:n2]

            data = torch.cat((b_data, o_data), 0)
            target = torch.cat((b_target, o_target), 0)
            domain_target = torch.cat((b_domain, o_domain), 0)

            target = target.type(torch.LongTensor)

            if cuda:
                data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

            data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

            if list(data.size())[0] == args.batch_size:
                batches += 1
                features = encoder(data)
                output_pred = regressor(features)

                op_0 = output_pred[:n1]
                target_0 = target[:n1]
                loss_0= criteron(op_0, target_0)

                op_1 = output_pred[n1:]
                target_1 = target[n1:]
                loss_1 = criteron(op_1, target_1)

                loss = loss_0 + loss_1
                val_loss += loss

                domains = domain_predictor.forward(features)
                domains = np.argmax(domains.detach().cpu().numpy(), axis=1)
                domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
                true_domains.append(domain_target)
                pred_domains.append(domains)

    val_loss = val_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)

    val_acc = accuracy_score(true_domains, pred_domains)

    print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss,  flush=True))
    print('Validation set: Average Acc: {:.4f}\n'.format(val_acc,  flush=True))

    return val_loss, val_acc












