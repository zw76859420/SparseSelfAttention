__author__ = "Jie Lei"

import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tqdm import tqdm
from tensorboardX import SummaryWriter

from model.tvqa_abc import ABC, baseline_model, selfattention_model, sparse_selfattention_model, sparse_selfattention_simple_model, baseline_model2, baseline_model_simple
from model.tvqa_model_ksc import RNN_simple, selfattention_simple, sparse_selfattention_simple, CNN_simple
from tvqa_dataset_origin import TVQADataset, pad_collate, preprocess_inputs
from config import BaseOptions
from utils import kl_categorical, kl_bernoulli, kl_categorical_reverse, kl_bernoulli_reverse#, kl_bernoulli_1
import numpy as np

#torch.autograd.set_detect_anomaly(True)


def train(opt, dset, model, criterion, log_prior, optimizer, epoch, previous_best_acc):
    dset.set_mode("train")
    model.train()
    train_loader = DataLoader(dset, batch_size=opt.bsz, shuffle=True, collate_fn=pad_collate)

    train_loss = []
    train_loss_kl = []
    valid_acc_log = ["batch_idx\tacc"]
    train_corrects = []
    torch.set_grad_enabled(True)
    for batch_idx, batch in tqdm(enumerate(train_loader)):
        model_inputs, targets, _ = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l,
                                                     device=opt.device)
        sub_l = model_inputs[-5]

        outputs, prob, _, _ = model(*model_inputs)
        #print("output", outputs)
        if opt.prior=="categorical":
            #print("categorical_here")
            #print(prob.sum())
            #loss_kl = kl_categorical(prob, log_prior, sub_l)
            loss_kl = kl_categorical_reverse(prob, log_prior, sub_l)
            #print(loss_kl.sum())
        elif opt.prior=="bernoulli":
            #print(prob.size(1)*prob.size(2)*prob.size(3))
            loss_kl = kl_bernoulli(prob, log_prior, opt.multihead, sub_l)
            #loss_kl = kl_bernoulli_reverse(prob, log_prior, opt.multihead, sub_l)
            #loss_kl = kl_bernoulli_reverse(prob, log_prior, opt.multihead, sub_l)
            #print(loss_kl)
        #elif opt.prior=="bernoulli_1":
            #loss_kl = kl_bernoulli_1(prob, log_prior, sub_l)
        else:
            loss_kl = 0.
        loss = criterion(outputs, targets) + opt.lambda_val * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        train_loss.append(loss.item())
        if opt.prior == "categorical" or opt.prior=="bernoulli" or opt.prior == "bernoulli_1":
            train_loss_kl.append(loss_kl.item())
        else:
            train_loss_kl.append(loss_kl)
        pred_ids = outputs.data.max(1)[1]
        train_corrects += pred_ids.eq(targets.data).cpu().numpy().tolist()
        if batch_idx % opt.log_freq == 0:
            niter = epoch * len(train_loader) + batch_idx

            train_acc = sum(train_corrects) / float(len(train_corrects))
            train_loss = sum(train_loss) / float(len(train_corrects))
            train_loss_kl = sum(train_loss_kl) / float(len(train_corrects))
            opt.writer.add_scalar("Train/Acc", train_acc, niter)
            opt.writer.add_scalar("Train/Loss", train_loss, niter)
            opt.writer.add_scalar("Train/Loss_kl", train_loss_kl, niter)

            # Test
            valid_acc, valid_loss, valid_loss_kl = validate(opt, dset, model, mode="valid")
            opt.writer.add_scalar("Valid/Loss", valid_loss, niter)
            opt.writer.add_scalar("Valid/Loss_kl", valid_loss_kl, niter)
            opt.writer.add_scalar("Valid/Acc", valid_acc, niter)

            valid_log_str = "%02d\t%.4f" % (batch_idx, valid_acc)
            valid_acc_log.append(valid_log_str)
            if valid_acc > previous_best_acc:
                previous_best_acc = valid_acc
                torch.save(model.state_dict(), os.path.join(opt.results_dir, "best_valid.pth"))
            print(" Train Epoch %d loss %.4f acc %.4f Val loss %.4f acc %.4f"
                  % (epoch, train_loss, train_acc, valid_loss, valid_acc))

            # reset to train
            torch.set_grad_enabled(True)
            model.train()
            dset.set_mode("train")
            train_corrects = []
            train_loss = []
            train_loss_kl = []

        if opt.debug:
            break

    # additional log
    with open(os.path.join(opt.results_dir, "valid_acc.log"), "a") as f:
        f.write("\n".join(valid_acc_log) + "\n")

    return previous_best_acc


def validate(opt, dset, model, mode="valid"):
    dset.set_mode(mode)
    torch.set_grad_enabled(False)
    model.eval()
    valid_loader = DataLoader(dset, batch_size=opt.test_bsz, shuffle=False, collate_fn=pad_collate)

    valid_qids = []
    valid_loss = []
    valid_corrects = []
    valid_loss_kl = []
    for _, batch in enumerate(valid_loader):
        model_inputs, targets, qids = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l,
                                                        device=opt.device)
        sub_l = model_inputs[-5]

        outputs, prob, _, _ = model(*model_inputs)
        if opt.prior == "categorical":
            #loss_kl = kl_categorical(prob, log_prior, sub_l)
            loss_kl = kl_categorical_reverse(prob, log_prior, sub_l)
        elif opt.prior == 'bernoulli':
            #loss_kl = kl_bernoulli(prob, log_prior, prob.size(1) * prob.size(2) * prob.size(3))
            loss_kl = kl_bernoulli(prob, log_prior, opt.multihead, sub_l)
            #loss_kl = kl_bernoulli_reverse(prob, log_prior, opt.multihead, sub_l)
            #loss_kl = kl_bernoulli_reverse(prob, log_prior, opt.multihead, sub_l)
        #elif opt.prior=="bernoulli_1":
        #    loss_kl = kl_bernoulli_1(prob, log_prior, sub_l)
        else:
            loss_kl = 0.
        loss = criterion(outputs, targets) + opt.lambda_val * loss_kl

        # measure accuracy and record loss
        valid_qids += [int(x) for x in qids]
        valid_loss.append(loss.item())
        if opt.prior == "bernoulli" or opt.prior=="categorical" or opt.prior == "bernoulli_1":
            valid_loss_kl.append(loss_kl.item())
        else:
            valid_loss_kl.append(loss_kl)
        pred_ids = outputs.data.max(1)[1]
        valid_corrects += pred_ids.eq(targets.data).cpu().numpy().tolist()

        if opt.debug:
            break

    valid_acc = sum(valid_corrects) / float(len(valid_corrects))
    valid_loss = sum(valid_loss) / float(len(valid_corrects))
    valid_loss_kl = sum(valid_loss_kl) / float(len(valid_corrects))
    return valid_acc, valid_loss, valid_loss_kl


if __name__ == "__main__":
    torch.manual_seed(2018)
    opt = BaseOptions().parse()
    writer = SummaryWriter(opt.results_dir)
    opt.writer = writer

    dset = TVQADataset(opt)
    opt.vocab_size = len(dset.word2idx)

    #model = RNN_simple(opt)
    #model = CNN_simple(opt)
    #model = selfattention_simple(opt)
    model = sparse_selfattention_simple(opt)
    if not opt.no_glove:
        model.load_embedding(dset.vocab_embedding)

    model.to(opt.device)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss(size_average=False).to(opt.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=opt.lr, weight_decay=opt.wd)

    if opt.prior=="categorical":
        prior = torch.ones(opt.multihead)/opt.multihead
        print("Using prior", prior)
        #print(prior)
        #log_prior = torch.log(prior)
        log_prior = prior
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = torch.unsqueeze(log_prior, -1)
        log_prior = torch.unsqueeze(log_prior, -1)
        #log_prior = Variable(log_prior)
        log_prior = Variable(log_prior).cuda()
    elif opt.prior=="bernoulli":

        log_prior = torch.Tensor([opt.ber_prior])
        print("Using prior", log_prior)
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = torch.unsqueeze(log_prior, -1)
        log_prior = torch.unsqueeze(log_prior, -1)
        # log_prior = Variable(log_prior)
        log_prior = Variable(log_prior).cuda()

    else:
        log_prior = 0
    best_acc = 0.
    early_stopping_cnt = 0
    early_stopping_flag = False
    for epoch in range(opt.n_epoch):
        if not early_stopping_flag:
            # train for one epoch, valid per n batches, save the log and the best model
            cur_acc = train(opt, dset, model, criterion, log_prior, optimizer, epoch, best_acc)

            # remember best acc
            is_best = cur_acc > best_acc
            best_acc = max(cur_acc, best_acc)
            if not is_best:
                early_stopping_cnt += 1
                if early_stopping_cnt >= opt.max_es_cnt:
                    early_stopping_flag = True
        else:
            print("early stop with valid acc %.4f" % best_acc)
            opt.writer.export_scalars_to_json(os.path.join(opt.results_dir, "all_scalars.json"))
            opt.writer.close()
            break  # early stop break

        if opt.debug:
            break



