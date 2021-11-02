__author__ = "Jie Lei"

import os
import json
import pickle
import torch

def read_json_lines(file_path):
    with open(file_path, "r") as f:
        lines = []
        for l in f.readlines():
            loaded_l = json.loads(l.strip("\n"))
            lines.append(loaded_l)
    return lines


def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)


def save_json_pretty(data, file_path):
    """save formatted json, use this one for some json config files"""
    with open(file_path, "w") as f:
        f.write(json.dumps(data, indent=4, sort_keys=True))


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_pickle(data, data_path):
    with open(data_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def files_exist(filepath_list):
    """check whether all the files exist"""
    for ele in filepath_list:
        if not os.path.exists(ele):
            return False
    return True


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    #kl_div = preds * (torch.log(preds + eps) - log_prior)
    input_mask = torch.arange(max(num_atoms)).expand(len(num_atoms), max(num_atoms)).cuda() >= num_atoms.unsqueeze(
        1)  # [B,T]

    input_mask_ = input_mask.unsqueeze(1).unsqueeze(-1)
    kl_div = (preds + eps) * (torch.log(preds + eps)-torch.log(log_prior))# - torch.log(log_prior)) + (1 - preds + eps) * (
    #            torch.log(1 - preds + eps) - torch.log(1 - log_prior))
    kl_div.data.masked_fill_(input_mask_.data.bool(), 0.)
    kl_div = kl_div.sum(dim=1)
    kl_div = kl_div.sum(dim=(1, 2)) / (num_atoms.float()**2)
    return kl_div.sum() / (preds.size(0))

def kl_categorical_reverse(preds, log_prior, num_atoms, eps=1e-16):
    #kl_div = preds * (torch.log(preds + eps) - log_prior)
    input_mask = torch.arange(max(num_atoms)).expand(len(num_atoms), max(num_atoms)).cuda() >= num_atoms.unsqueeze(
        1)  # [B,T]

    input_mask_ = input_mask.unsqueeze(1).unsqueeze(-1)
    kl_div = -log_prior * torch.log(preds+eps)
    kl_div.data.masked_fill_(input_mask_.data.bool(), 0.)
    kl_div = kl_div.sum(dim=1)
    kl_div = kl_div.sum(dim=(1, 2)) / (num_atoms.float()**2)
    return kl_div.sum() / (preds.size(0))

def kl_bernoulli(preds, log_prior, num_head, num_atoms, eps=1e-16):

    input_mask = torch.arange(max(num_atoms)).expand(len(num_atoms), max(num_atoms)).cuda() >= num_atoms.unsqueeze(
        1)  # [B,T]

    input_mask_ = input_mask.unsqueeze(1).unsqueeze(-1)


    kl_div = (preds+eps)*(torch.log(preds+eps)-torch.log(log_prior)) + (1-preds+eps)*(torch.log(1-preds+eps)-torch.log(1-log_prior))
    kl_div.data.masked_fill_(input_mask_.data.bool(), 0.)
    kl_div = kl_div.sum(dim=1) / num_head
    kl_div = kl_div.sum(dim=(1, 2)) / (num_atoms.float()**2)
    #print("KL_div: ", kl_div)
    return kl_div.sum() / (preds.size(0))

def kl_bernoulli_reverse(preds, log_prior, num_head, num_atoms, eps=1e-16):

    input_mask = torch.arange(max(num_atoms)).expand(len(num_atoms), max(num_atoms)).cuda() >= num_atoms.unsqueeze(
        1)  # [B,T]

    input_mask_ = input_mask.unsqueeze(1).unsqueeze(-1)


    kl_div = (log_prior)*(torch.log(log_prior)-torch.log(preds+eps)) + (1-log_prior)*(torch.log(1-log_prior)-torch.log(1-preds+eps))
    kl_div.data.masked_fill_(input_mask_.data.bool(), 0.)
    kl_div = kl_div.sum(dim=1) / num_head
    kl_div = kl_div.sum(dim=(1, 2)) / (num_atoms.float()**2)
    #print("KL_div: ", kl_div)
    return kl_div.sum() / (preds.size(0))

