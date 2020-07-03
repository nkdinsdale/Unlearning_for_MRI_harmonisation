# Nicola Dinsdale 2020
# Functions for training and validating the model
########################################################################################################################
# Import dependencies
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
########################################################################################################################

def train_unlearn_threedatasets(args, models, train_loaders, optimizers, criterions, epoch):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [optimizer, optimizer_conf, optimizer_dm] = optimizers
    [b_train_dataloader, o_train_dataloader, w_train_dataloader] = train_loaders
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
    for batch_idx, ((b_data, b_target, b_domain), (o_data, o_target, o_domain), (w_data, w_target, w_domain)) in enumerate(zip(b_train_dataloader, o_train_dataloader, w_train_dataloader)):
        max_batch = len(b_data)
        n1 = np.random.randint(1, max_batch - 2)        # Must be at least one from each
        n2 = np.random.randint(1, max_batch - n1 -1)
        n3 = max_batch - n1 - n2
        if n3 < 1:
            assert ValueError('N3 must be greater that zero')

        b_data = b_data[:n1]
        b_target = b_target[:n1]
        b_domain = b_domain[:n1]

        o_data = o_data[:n2]
        o_target = o_target[:n2]
        o_domain = o_domain[:n2]

        w_data = w_data[:n3]
        w_target = w_target[:n3]
        w_domain = w_domain[:n3]

        data = torch.cat((b_data, o_data, w_data), 0)
        target = torch.cat((b_target, o_target, w_target), 0)
        domain_target = torch.cat((b_domain, o_domain, w_domain), 0)

        if cuda:
            data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

        data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

        if list(data.size())[0] == args.batch_size :
            batches += 1

            # First update the encoder and regressor
            optimizer.zero_grad()
            features = encoder(data)
            output_pred = regressor(features)
            loss_1 = criteron(output_pred[:n1], target[:n1])
            loss_2 = criteron(output_pred[n1:n1+n2], target[n1:n1+n2])
            loss_3 = criteron(output_pred[n1+n2:], target[n1+n2:])
            loss = loss_1 + loss_2 + loss_3
            loss_total = loss
            loss_total.backward(retain_graph=True)
            optimizer.step()

            # Now update just the domain classifier
            optimizer_dm.zero_grad()
            output_dm = domain_predictor(features.detach())
            loss_dm = args.alpha * domain_criterion(output_dm, domain_target)
            loss_dm.backward()
            optimizer_dm.step()

            # Now update just the encoder using the domain loss
            optimizer_conf.zero_grad()
            output_dm_conf = domain_predictor(features)
            loss_conf = args.beta * conf_criterion(output_dm_conf, domain_target)        # Get rid of the weight for not unsupervised
            loss_conf.backward(retain_graph=False)
            optimizer_conf.step()

            regressor_loss += loss
            domain_loss += loss_dm
            conf_loss += loss_conf

            output_dm_conf = np.argmax(output_dm_conf.detach().cpu().numpy(), axis=1)
            domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
            true_domains.append(domain_target)
            pred_domains.append(output_dm_conf)

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx+1) * len(data), len(b_train_dataloader.dataset),
                           100. * (batch_idx+1) / len(b_train_dataloader), loss.item()), flush=True)
                print('\t \t Confusion loss = ', loss_conf.item())
                print('\t \t Domain Loss = ', loss_dm.item(), flush=True)
            del target
            del loss
            del features

    av_loss = regressor_loss / batches
    av_conf = conf_loss / batches
    av_dom = domain_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss,  flush=True))
    print('Training set: Average Conf loss: {:.4f}'.format(av_conf,  flush=True))
    print('Training set: Average Dom loss: {:.4f}'.format(av_dom,  flush=True))
    print('Training set: Average Acc: {:.4f}\n'.format(acc,  flush=True))

    return av_loss, acc, av_dom, av_conf

def train_unlearn_distinct(args, models, train_loaders, optimizers, criterions, epoch):
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
                loss_1 = criteron(output_pred[:n1], target[:n1])
                loss_2 = criteron(output_pred[n1:], target[n1:])
                loss_total = loss_1 + loss_2
                loss_total.backward()
                optimizer.step()

                # Now update just the domain classifier on the intersection data only
                optimizer_dm.zero_grad()
                new_features = encoder(int_data)
                output_dm = domain_predictor(new_features.detach())
                loss_dm = args.alpha * domain_criterion(output_dm, int_domain)
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
    av_conf = conf_loss / batches
    av_dom = domain_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)

    acc = accuracy_score(true_domains, pred_domains)

    print('Training set: Average loss: {:.4f}'.format(av_loss,  flush=True))
    print('Training set: Average Conf loss: {:.4f}'.format(av_conf,  flush=True))
    print('Training set: Average Dom loss: {:.4f}'.format(av_dom,  flush=True))

    print('Training set: Average Acc: {:.4f}\n'.format(acc,  flush=True))

    return av_loss, acc, av_dom, av_conf

def val_unlearn_threedatasets(args, models, val_loaders, criterions):

    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [b_val_dataloader, o_val_dataloader, w_val_dataloader] = val_loaders
    [criteron, _, _] = criterions

    encoder.eval()
    regressor.eval()
    domain_predictor.eval()

    val_loss = 0

    true_domains = []
    pred_domains = []

    batches = 0
    with torch.no_grad():
        for batch_idx, ((b_data, b_target, b_domain), (o_data, o_target, o_domain), (w_data, w_target, w_domain)) in enumerate(zip(b_val_dataloader, o_val_dataloader, w_val_dataloader)):
            max_batch = len(b_data)
            n1 = np.random.randint(1, max_batch - 2)  # Must be at least one from each
            n2 = np.random.randint(1, max_batch - n1 - 1)
            n3 = max_batch - n1 - n2
            if n3 < 1:
                assert ValueError('N3 must be greater that zero')

            b_data = b_data[:n1]
            b_target = b_target[:n1]
            b_domain = b_domain[:n1]

            o_data = o_data[:n2]
            o_target = o_target[:n2]
            o_domain = o_domain[:n2]

            w_data = w_data[:n3]
            w_target = w_target[:n3]
            w_domain = w_domain[:n3]

            data = torch.cat((b_data, o_data, w_data), 0)
            target = torch.cat((b_target, o_target, w_target), 0)
            domain_target = torch.cat((b_domain, o_domain, w_domain), 0)

            if cuda:
                data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

            data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

            if list(data.size())[0] == args.batch_size:
                batches += 1
                features = encoder(data)
                output_pred = regressor(features)
                loss_1 = criteron(output_pred[:n1], target[:n1])
                loss_2 = criteron(output_pred[n1:n1+n2], target[n1:n1+n2])
                loss_3 = criteron(output_pred[n1+n2:n1+n2+n3], target[n1+n2:n1+n2+n3])
                loss = loss_1 + loss_2 + loss_3
                val_loss += loss

                domains = domain_predictor.forward(features)
                domains = np.argmax(domains.detach().cpu().numpy(), axis=1)
                domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
                true_domains.append(domain_target)
                pred_domains.append(domains)

    val_loss = val_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss,  flush=True))
    print('Validation set: Average Acc: {:.4f}\n'.format(acc,  flush=True))

    return val_loss, acc

def val_unlearn_distinct(args, models, val_loaders, criterions):

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

            if cuda:
                data, target, domain_target, int_data, int_domain = data.cuda(), target.cuda(), domain_target.cuda(), int_data.cuda(), int_domain.cuda()

            data, target, domain_target, int_data, int_domain = Variable(data), Variable(target), Variable(domain_target), Variable(int_data), Variable(int_domain)

            if list(data.size())[0] == args.batch_size:
                if list(int_data.size())[0] == args.batch_size:
                    batches += 1
                    features = encoder(data)
                    output_pred = regressor(features)
                    loss_1 = criteron(output_pred[:n1], target[:n1])
                    loss_2 = criteron(output_pred[n1:n1+n2], target[n1:n1+n2])
                    loss_3 = criteron(output_pred[n1+n2:], target[n1+n2:])
                    loss = loss_1 + loss_2 + loss_3
                    val_loss += loss

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

def train_encoder_unlearn_threedatasets(args, models, train_loaders, optimizers, criterions, epoch):
    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [optimizer] = optimizers
    [b_train_dataloader, o_train_dataloader, w_train_dataloader] = train_loaders
    [criteron, _, domain_criterion] = criterions

    regressor_loss = 0
    domain_loss = 0

    encoder.train()
    regressor.train()
    domain_predictor.train()

    true_domains = []
    pred_domains = []

    batches = 0
    for batch_idx, ((b_data, b_target, b_domain), (o_data, o_target, o_domain), (w_data, w_target, w_domain)) in enumerate(zip(b_train_dataloader, o_train_dataloader, w_train_dataloader)):
        max_batch = len(b_data)
        n1 = np.random.randint(1, max_batch - 2)  # Must be at least one from each
        n2 = np.random.randint(1, max_batch - n1 - 1)
        n3 = max_batch - n1 - n2
        if n3 < 1:
            assert ValueError('N3 must be greater that zero')

        b_data = b_data[:n1]
        b_target = b_target[:n1]
        b_domain = b_domain[:n1]

        o_data = o_data[:n2]
        o_target = o_target[:n2]
        o_domain = o_domain[:n2]

        w_data = w_data[:n3]
        w_target = w_target[:n3]
        w_domain = w_domain[:n3]

        data = torch.cat((b_data, o_data, w_data), 0)
        target = torch.cat((b_target, o_target, w_target), 0)
        domain_target = torch.cat((b_domain, o_domain, w_domain), 0)

        if cuda:
            data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

        data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

        if list(data.size())[0] == args.batch_size :
            batches += 1

            # First update the encoder and regressor
            optimizer.zero_grad()
            features = encoder(data)
            output_pred = regressor(features)
            domain_pred = domain_predictor(features)
            loss_1 = criteron(output_pred[:n1], target[:n1])
            loss_2 = criteron(output_pred[n1:n1+n2], target[n1:n1+n2])
            loss_3 = criteron(output_pred[n1+n2:], target[n1+n2:])
            r_loss = loss_1 + loss_2 + loss_3
            d_loss = domain_criterion(domain_pred, domain_target)
            loss = r_loss + args.alpha * d_loss
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

def train_encoder_domain_unlearn_distinct(args, models, train_loaders, optimizers, criterions, epoch):
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

        if cuda:
            data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

        data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

        if list(data.size())[0] == args.batch_size :

            batches += 1

            # First update the encoder and regressor for now dont improve the domain stuff, just the feature predictions
            optimizer.zero_grad()
            features = encoder(data)
            output_pred = regressor(features)
            loss_1 = criteron(output_pred[:n1], target[:n1])
            loss_2 = criteron(output_pred[n1:], target[n1:])
            loss = loss_1 + loss_2
            regressor_loss += loss

            output_dm = domain_predictor(features.detach())
            loss_dm = domain_criterion(output_dm, domain_target)

            loss = loss + loss_dm
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

    av_dom = domain_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss,  flush=True))
    print('Training set: Average Dom loss: {:.4f}'.format(av_dom,  flush=True))
    print('Training set: Average Acc: {:.4f}\n'.format(acc,  flush=True))

    return av_loss, acc, av_dom, np.NaN

def val_encoder_unlearn_threedatasets(args, models, val_loaders, criterions):

    cuda = torch.cuda.is_available()

    [encoder, regressor, domain_predictor] = models
    [b_val_dataloader, o_val_dataloader, w_val_dataloader] = val_loaders
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
        for batch_idx, ((b_data, b_target, b_domain), (o_data, o_target, o_domain), (w_data, w_target, w_domain)) in enumerate(zip(b_val_dataloader, o_val_dataloader, w_val_dataloader)):
            max_batch = len(b_data)
            n1 = np.random.randint(1, max_batch - 2)  # Must be at least one from each
            n2 = np.random.randint(1, max_batch - n1 - 1)
            n3 = max_batch - n1 - n2
            if n3 < 1:
                assert ValueError('N3 must be greater that zero')

            b_data = b_data[:n1]
            b_target = b_target[:n1]
            b_domain = b_domain[:n1]

            o_data = o_data[:n2]
            o_target = o_target[:n2]
            o_domain = o_domain[:n2]

            w_data = w_data[:n3]
            w_target = w_target[:n3]
            w_domain = w_domain[:n3]

            data = torch.cat((b_data, o_data, w_data), 0)
            target = torch.cat((b_target, o_target, w_target), 0)
            domain_target = torch.cat((b_domain, o_domain, w_domain), 0)

            if cuda:
                data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

            data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

            if list(data.size())[0] == args.batch_size:
                batches += 1
                features = encoder(data)
                output_pred = regressor(features)
                domain_pred = domain_predictor(features)

                loss_1 = criteron(output_pred[:n1], target[:n1])
                loss_2 = criteron(output_pred[n1:n1+n2], target[n1:n1+n2])
                loss_3 = criteron(output_pred[n1+n2:], target[n1+n2:])

                r_loss =  loss_1 + loss_2 + loss_3
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

def val_encoder_domain_unlearn_distinct(args, models, val_loaders, criterions):

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

            if cuda:
                data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

            data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

            if list(data.size())[0] == args.batch_size:
                batches += 1
                features = encoder(data)
                output_pred = regressor(features)
                loss_1 = criteron(output_pred[:n1], target[:n1])
                loss_2 = criteron(output_pred[n1:], target[n1:] )
                loss = loss_1 + loss_2
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

    print('\nValidation set: Average loss: {:.4f}'.format(val_loss,  flush=True))
    print('Validation set: Average Acc: {:.4f}\n'.format(val_acc,  flush=True))

    return val_loss, val_acc

