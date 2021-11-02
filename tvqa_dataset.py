__author__ = "Jie Lei"

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from utils import load_pickle, save_pickle, load_json, files_exist
from pytorch_pretrained_bert import BertTokenizer, BertModel


class TVQADataset(Dataset):
    def __init__(self, opt, mode="train"):
        self.raw_train = load_json(opt.train_path)
        self.raw_test = load_json(opt.test_path)
        self.raw_valid = load_json(opt.valid_path)
        self.vcpt_dict = load_pickle(opt.vcpt_path)
        self.vfeat_load = opt.vid_feat_flag
        if self.vfeat_load:
            self.vid_h5 = h5py.File(opt.vid_feat_path, "r", driver=opt.h5driver)
        self.glove_embedding_path = opt.glove_path
        self.normalize_v = opt.normalize_v
        self.with_ts = opt.with_ts
        self.mode = mode
        self.cur_data_dict = self.get_cur_dict()

        # set BERT embedding
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()

        # set entry keys
        if self.with_ts:
            self.text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "located_sub_text"]
        else:
            self.text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub_text"]
        self.vcpt_key = "vcpt"
        self.label_key = "answer_idx"
        self.qid_key = "qid"
        self.vid_name_key = "vid_name"
        self.located_frm_key = "located_frame"
        for k in self.text_keys + [self.vcpt_key, self.qid_key, self.vid_name_key]:
            if k == "vcpt":
                continue
            assert k in self.raw_valid[0].keys()

    def set_mode(self, mode):
        self.mode = mode
        self.cur_data_dict = self.get_cur_dict()

    def get_cur_dict(self):
        if self.mode == 'train':
            return self.raw_train
        elif self.mode == 'valid':
            return self.raw_valid
        elif self.mode == 'test':
            return self.raw_test

    def __len__(self):
        return len(self.cur_data_dict)

    def __getitem__(self, index):
        items = []
        if self.with_ts:
            cur_start, cur_end = self.cur_data_dict[index][self.located_frm_key]
        cur_vid_name = self.cur_data_dict[index][self.vid_name_key]

        # add text keys
        for k in self.text_keys:
            # items.append(self.numericalize(self.cur_data_dict[index][k]))
            items.append(self.embedding_BERT(self.cur_data_dict[index][k]))

        ########### vcpt embedding?? ###########

        # # add vcpt
        # if self.with_ts:
        #     cur_vis_sen = self.vcpt_dict[cur_vid_name][cur_start:cur_end + 1]
        # else:
        #     cur_vis_sen = self.vcpt_dict[cur_vid_name]
        # cur_vis_sen = " , ".join(cur_vis_sen)
        # items.append(self.numericalize_vcpt(cur_vis_sen))

        # add other keys
        if self.mode == 'test':
            items.append(666)  # this value will not be used
        else:
            items.append(int(self.cur_data_dict[index][self.label_key]))
        for k in [self.qid_key]:
            items.append(self.cur_data_dict[index][k])
        items.append(cur_vid_name)

        # add visual feature
        if self.vfeat_load:
            if self.with_ts:
                cur_vid_feat = torch.from_numpy(self.vid_h5[cur_vid_name][cur_start:cur_end])
            else:  # handled by vid_path
                cur_vid_feat = torch.from_numpy(self.vid_h5[cur_vid_name][:480])
            if self.normalize_v:
                cur_vid_feat = nn.functional.normalize(cur_vid_feat, p=2, dim=1)
        else:
            cur_vid_feat = torch.zeros([2, 2])  # dummy placeholder
        items.append(cur_vid_feat)
        return items

    # ########### vcpt embedding change ??? ##############
    # def numericalize_vcpt(self, vcpt_sentence):
    #     """convert words to indices, additionally removes duplicated attr-object pairs"""
    #     attr_obj_pairs = vcpt_sentence.lower().split(",")  # comma is also removed
    #     unique_pairs = []
    #     for pair in attr_obj_pairs:
    #         if pair not in unique_pairs:
    #             unique_pairs.append(pair)
    #     words = []
    #     for pair in unique_pairs:
    #         words.extend(pair.split())
    #     words.append("<eos>")
    #     sentence_indices = [self.word2idx[w] if w in self.word2idx else self.word2idx["<unk>"]
    #                         for w in words]
    #     return sentence_indices

    def embedding_BERT(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        segments_ids = []
        sentence_idx = 0
        for token in tokenized_text:
            segments_ids.append(sentence_idx)
            if token == '.' or token == '?' or token == '!':
                sentence_idx += 1

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
        ########### how to sum? many BERT result matrix ###########
        # 12 len list, [1, sentence_len, 768]
        sentence_len = len(segments_ids)
        layer_sum = torch.zeros([1, len(segments_ids), 768])
        for layer in encoded_layers:
            layer_sum += layer

        return layer_sum/sentence_len


class Batch(object):
    def __init__(self):
        self.__doc__ = "empty initialization"

    @classmethod
    def get_batch(cls, keys=None, values=None):
        """Create a Batch directly from a number of Variables."""
        batch = cls()
        assert keys is not None and values is not None
        for k, v in zip(keys, values):
            setattr(batch, k, v)
        return batch


def pad_collate(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq)."""
    # padding
    def pad_video_sequences(sequences):
        """sequences is a list of torch float tensors (created from numpy)"""
        lengths = torch.LongTensor([len(seq) for seq in sequences])
        v_dim = sequences[0].size(1)
        padded_seqs = torch.zeros(len(sequences), max(lengths), v_dim).float()
        for idx, seq in enumerate(sequences):
            end = lengths[idx]
            padded_seqs[idx, :end] = seq
        return padded_seqs, lengths

    # separate source and target sequences
    column_data = list(zip(*data))
    text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub"] #, "vcpt"
    label_key = "answer_idx"
    qid_key = "qid"
    # vid_name_key = "vid_name"
    # vid_feat_key = "vid"
    all_keys = text_keys + [label_key, qid_key]
    all_values = []
    for i, k in enumerate(all_keys):
        if k in text_keys:
            all_values.append(column_data[i])
        elif k == label_key:
            all_values.append(torch.LongTensor(column_data[i]))
        # elif k == vid_feat_key:
        #     all_values.append(pad_video_sequences(column_data[i]))
        # else:
        #     all_values.append(column_data[i])

    batched_data = Batch.get_batch(keys=all_keys, values=all_values)
    return batched_data


def preprocess_inputs(batched_data, max_sub_l, max_vcpt_l, max_vid_l, device="cuda:0"):
    """clip and move to target device"""
    max_len_dict = {"sub": max_sub_l, "vcpt": max_vcpt_l, "vid": max_vid_l}
    text_keys = ["q", "a0", "a1", "a2", "a3", "a4", "sub", "vcpt"]
    label_key = "answer_idx"
    qid_key = "qid"
    vid_feat_key = "vid"
    model_in_list = []
    for k in text_keys + [vid_feat_key]:
        v = getattr(batched_data, k)
        if k in max_len_dict:
            ctx, ctx_l = v
            max_l = min(ctx.size(1), max_len_dict[k])
            if ctx.size(1) > max_l:
                ctx_l = ctx_l.clamp(min=1, max=max_l)
                ctx = ctx[:, :max_l]
            model_in_list.extend([ctx.to(device), ctx_l.to(device)])
        else:
            model_in_list.extend([v[0].to(device), v[1].to(device)])
    target_data = getattr(batched_data, label_key)
    target_data = target_data.to(device)
    qid_data = getattr(batched_data, qid_key)
    return model_in_list, target_data, qid_data


if __name__ == "__main__":
    # python tvqa_dataset.py --input_streams sub
    import sys
    from config import BaseOptions
    sys.argv[1:] = ["--input_streams", "sub"]
    opt = BaseOptions().parse()

    dset = TVQADataset(opt, mode="valid")
    print(dset.__getitem__(100))

    data_loader = DataLoader(dset, batch_size=5, shuffle=False, collate_fn=pad_collate)

    for batch_idx, batch in enumerate(data_loader):
        print(batch)
        # model_inputs, targets, qids = preprocess_inputs(batch, opt.max_sub_l, opt.max_vcpt_l, opt.max_vid_l)
        # print(model_inputs, targets, qids)
        break

# Todo: embedding단에서 runtime error: index out of range(index 접근 잘못?)
# get_item은 실행됨, pad_collate단에서 error