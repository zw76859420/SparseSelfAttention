__author__ = "Jie Lei"

import torch
from torch import nn

from model.rnn import RNNEncoder, max_along_time, mean_along_time
from model.bidaf import BidafAttn
from model.mlp import MLP
from model.selfattention_test import SelfAttention
from model.position_encoding import PositionEncoding

class selfattention_model(nn.Module):
    def __init__(self, opt):
        super(selfattention_model, self).__init__()
        self.vid_flag = "imagenet" in opt.input_streams
        self.sub_flag = "sub" in opt.input_streams
        self.vcpt_flag = "vcpt" in opt.input_streams
        #hidden_size_1 = opt.hsz1
        hidden_size = opt.hsz2
        n_layers_cls = opt.n_layers_cls
        vid_feat_size = opt.vid_feat_size
        embedding_size = opt.embedding_size
        vocab_size = opt.vocab_size
        head_num = opt.multihead
        self.position_encoding = PositionEncoding(n_filters=embedding_size)

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bidaf = BidafAttn(hidden_size * 3, method="dot")  # no parameter for dot
        self.seq_raw = SelfAttention(hidden_size, hidden_size, dropout_p=0.,
                                     attention_head=head_num)

        self.merge = nn.Sequential(
            nn.Linear(embedding_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.),
        )
        self.merge_q = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.),
        )

        if self.vid_flag:
            print("activate video stream")
            self.video_fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(vid_feat_size, embedding_size),
                nn.Tanh(),
            )
            self.sequence_vid = SelfAttention(hidden_size, hidden_size, dropout_p=0.,
                                              attention_head=head_num)
            self.classifier_vid = MLP(hidden_size, 1, 500, n_layers_cls)

        if self.sub_flag:
            print("activate sub stream")
            self.sequence_sub = SelfAttention(hidden_size, hidden_size, dropout_p=0.,
                                              attention_head=head_num)
            self.classifier_sub = MLP(hidden_size, 1, 500, n_layers_cls)

        if self.vcpt_flag:
            print("activate vcpt stream")
            self.sequence_vcpt = SelfAttention(hidden_size, hidden_size, dropout_p=0.,
                                               attention_head=head_num)
            self.classifier_vcpt = MLP(hidden_size, 1, 500, n_layers_cls)

    def load_embedding(self, pretrained_embedding):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    def forward(self, q, q_l, a0, a0_l, a1, a1_l, a2, a2_l, a3, a3_l, a4, a4_l,
                sub, sub_l, vcpt, vcpt_l, vid, vid_l):
        e_q = self.embedding(q)
        e_a0 = self.embedding(a0)
        e_a1 = self.embedding(a1)
        e_a2 = self.embedding(a2)
        e_a3 = self.embedding(a3)
        e_a4 = self.embedding(a4)

        e_q = self.position_encoding(e_q)
        e_a0 = self.position_encoding(e_a0)
        e_a1 = self.position_encoding(e_a1)
        e_a2 = self.position_encoding(e_a2)
        e_a3 = self.position_encoding(e_a3)
        e_a4 = self.position_encoding(e_a4)

        # e_q, _ = self.seq_raw(e_q, q_l)
        # e_a0, _ = self.seq_raw(e_a0, a0_l)
        # e_a1, _ = self.seq_raw(e_a1, a1_l)
        # e_a2, _ = self.seq_raw(e_a2, a2_l)
        # e_a3, _ = self.seq_raw(e_a3, a3_l)
        # e_a4, _ = self.seq_raw(e_a4, a4_l)
        #
        if self.sub_flag:
            e_sub = self.embedding(sub)
            e_sub = self.position_encoding(e_sub)
            # e_sub, _ = self.seq_raw(e_sub, sub_l)
            #raw_out_sub, _ = self.lstm_raw(e_sub, sub_l)
            sub_out = self.stream_processor(self.sequence_sub, self.classifier_sub, e_sub, sub_l,
                                            e_q, q_l, e_a0, a0_l, e_a1, a1_l,
                                            e_a2, a2_l, e_a3, a3_l, e_a4, a4_l)
        else:
            sub_out = 0

        if self.vcpt_flag:
            e_vcpt = self.embedding(vcpt)
            e_vcpt = self.position_encoding(e_vcpt)
            # e_vcpt, _ = self.seq_raw(e_vcpt, vcpt_l)
            #raw_out_vcpt, _ = self.lstm_raw(e_vcpt, vcpt_l)
            vcpt_out = self.stream_processor(self.sequence_vcpt, self.classifier_vcpt, e_vcpt, vcpt_l,
                                             e_q, q_l, e_a0, a0_l, e_a1, a1_l,
                                             e_a2, a2_l, e_a3, a3_l, e_a4, a4_l)
        else:
            vcpt_out = 0

        if self.vid_flag:
            e_vid = self.video_fc(vid)
            e_vid = self.position_encoding(e_vid)
            # e_vid, _ = self.seq_raw(e_vid, vid_l)
            #raw_out_vid, _ = self.lstm_raw(e_vid, vid_l)
            vid_out = self.stream_processor(self.sequence_vid, self.classifier_vid, e_vid, vid_l,
                                            e_q, q_l, e_a0, a0_l, e_a1, a1_l,
                                            e_a2, a2_l, e_a3, a3_l, e_a4, a4_l)
        else:
            vid_out = 0

        out = sub_out + vcpt_out + vid_out  # adding zeros has no effect on backward
        return out.squeeze()

    def stream_processor(self, sequence_model, classifier, ctx_embed, ctx_l,
                         q_embed, q_l, a0_embed, a0_l, a1_embed, a1_l, a2_embed, a2_l, a3_embed, a3_l, a4_embed, a4_l):
        u_q, _ = self.bidaf(ctx_embed, ctx_l, q_embed, q_l)
        u_a0, _ = self.bidaf(ctx_embed, ctx_l, a0_embed, a0_l)
        u_a1, _ = self.bidaf(ctx_embed, ctx_l, a1_embed, a1_l)
        u_a2, _ = self.bidaf(ctx_embed, ctx_l, a2_embed, a2_l)
        u_a3, _ = self.bidaf(ctx_embed, ctx_l, a3_embed, a3_l)
        u_a4, _ = self.bidaf(ctx_embed, ctx_l, a4_embed, a4_l)


        query_a0 = torch.cat([u_q, u_a0], dim=-1)
        concat_a0 = torch.cat([ctx_embed, u_q * ctx_embed, u_a0 * ctx_embed], dim=-1)
        query_a1 = torch.cat([u_q, u_a1], dim=-1)
        concat_a1 = torch.cat([ctx_embed, u_q * ctx_embed, u_a1 * ctx_embed], dim=-1)
        query_a2 = torch.cat([u_q, u_a0], dim=-1)
        concat_a2 = torch.cat([ctx_embed, u_q * ctx_embed, u_a2 * ctx_embed], dim=-1)
        query_a3 = torch.cat([u_q, u_a3], dim=-1)
        concat_a3 = torch.cat([ctx_embed, u_q * ctx_embed, u_a3 * ctx_embed], dim=-1)
        query_a4 = torch.cat([u_q, u_a0], dim=-1)
        concat_a4 = torch.cat([ctx_embed, u_q * ctx_embed, u_a4 * ctx_embed], dim=-1)

        concat_a0 = self.merge(concat_a0)
        concat_a1 = self.merge(concat_a1)
        concat_a2 = self.merge(concat_a2)
        concat_a3 = self.merge(concat_a3)
        concat_a4 = self.merge(concat_a4)

        query_a0 = self.merge_q(query_a0)
        query_a1 = self.merge_q(query_a1)
        query_a2 = self.merge_q(query_a2)
        query_a3 = self.merge_q(query_a3)
        query_a4 = self.merge_q(query_a4)


        mature_maxout_a0, _ = sequence_model(query_a0, concat_a0, ctx_l)
        mature_maxout_a1, _ = sequence_model(query_a1, concat_a1, ctx_l)
        mature_maxout_a2, _ = sequence_model(query_a2, concat_a2, ctx_l)
        mature_maxout_a3, _ = sequence_model(query_a3, concat_a3, ctx_l)
        mature_maxout_a4, _ = sequence_model(query_a4, concat_a4, ctx_l)

        mature_maxout_a0 = mean_along_time(mature_maxout_a0, ctx_l).unsqueeze(1)
        mature_maxout_a1 = mean_along_time(mature_maxout_a1, ctx_l).unsqueeze(1)
        mature_maxout_a2 = mean_along_time(mature_maxout_a2, ctx_l).unsqueeze(1)
        mature_maxout_a3 = mean_along_time(mature_maxout_a3, ctx_l).unsqueeze(1)
        mature_maxout_a4 = mean_along_time(mature_maxout_a4, ctx_l).unsqueeze(1)

        mature_answers = torch.cat([
            mature_maxout_a0, mature_maxout_a1, mature_maxout_a2, mature_maxout_a3, mature_maxout_a4
        ], dim=1)
        out = classifier(mature_answers)  # (B, 5)
        return out
