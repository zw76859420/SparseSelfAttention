__author__ = "Jie Lei"

import torch
from torch import nn

from model.rnn import RNNEncoder, max_along_time, mean_along_time
from model.bidaf import BidafAttn
from model.mlp import MLP
from model.selfattention import SelfAttention, SparseSelfAttention
from model.position_encoding import PositionEncoding


class RNN_simple(nn.Module):
    def __init__(self, opt):
        super(RNN_simple, self).__init__()
        self.vid_flag = "imagenet" in opt.input_streams
        self.sub_flag = "sub" in opt.input_streams
        self.vcpt_flag = "vcpt" in opt.input_streams
        # hidden_size_1 = opt.hsz1
        hidden_size = opt.hsz2
        n_layers_cls = opt.n_layers_cls
        vid_feat_size = opt.vid_feat_size
        embedding_size = opt.embedding_size
        vocab_size = opt.vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bidaf = BidafAttn(hidden_size * 3, method="dot")  # no parameter for dot
        self.merge = nn.Sequential(
            nn.Linear(embedding_size * 5, hidden_size),
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
            self.lstm_mature_vid = RNNEncoder(hidden_size, hidden_size // 2, bidirectional=True,
                                              dropout_p=0, n_layers=1, rnn_type="gru")
            self.classifier_vid = MLP(hidden_size, 1, 500, n_layers_cls)

        if self.sub_flag:
            print("activate sub stream")
            self.lstm_mature_sub = RNNEncoder(hidden_size, hidden_size // 2, bidirectional=True,
                                              dropout_p=0, n_layers=1, rnn_type="gru")
            self.classifier_sub = MLP(hidden_size, 1, 500, n_layers_cls)

        if self.vcpt_flag:
            print("activate vcpt stream")
            self.lstm_mature_vcpt = RNNEncoder(hidden_size, hidden_size // 2, bidirectional=True,
                                               dropout_p=0, n_layers=1, rnn_type="gru")
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


        if self.sub_flag:
            e_sub = self.embedding(sub)
            sub_out = self.stream_processor(self.lstm_mature_sub, self.classifier_sub, e_sub, sub_l,
                                            e_q, q_l, e_a0, a0_l, e_a1, a1_l,
                                            e_a2, a2_l, e_a3, a3_l, e_a4, a4_l)
        else:
            sub_out = 0

        if self.vcpt_flag:
            e_vcpt = self.embedding(vcpt)
            vcpt_out = self.stream_processor(self.lstm_mature_vcpt, self.classifier_vcpt, e_vcpt, vcpt_l,
                                             e_q, q_l, e_a0, a0_l, e_a1, a1_l,
                                             e_a2, a2_l, e_a3, a3_l, e_a4, a4_l)
        else:
            vcpt_out = 0

        if self.vid_flag:
            e_vid = self.video_fc(vid)
            vid_out = self.stream_processor(self.lstm_mature_vid, self.classifier_vid, e_vid, vid_l,
                                            e_q, q_l, e_a0, a0_l, e_a1, a1_l,
                                            e_a2, a2_l, e_a3, a3_l, e_a4, a4_l)
        else:
            vid_out = 0
        edge_prob_sub = 0
        edge_prob_vcpt = 0
        edge_prob_vid = 0
        out = sub_out + vcpt_out + vid_out  # adding zeros has no effect on backward
        return out.squeeze(), edge_prob_sub, edge_prob_vcpt, edge_prob_vid




    def stream_processor(self, seq_model, classifier, ctx_embed, ctx_l,
                         q_embed, q_l, a0_embed, a0_l, a1_embed, a1_l, a2_embed, a2_l, a3_embed, a3_l, a4_embed, a4_l):

        ctx_embed, _ = seq_model(ctx_embed, ctx_l)

        u_q, _ = self.bidaf(ctx_embed, ctx_l, q_embed, q_l)
        u_a0, _ = self.bidaf(ctx_embed, ctx_l, a0_embed, a0_l)
        u_a1, _ = self.bidaf(ctx_embed, ctx_l, a1_embed, a1_l)
        u_a2, _ = self.bidaf(ctx_embed, ctx_l, a2_embed, a2_l)
        u_a3, _ = self.bidaf(ctx_embed, ctx_l, a3_embed, a3_l)
        u_a4, _ = self.bidaf(ctx_embed, ctx_l, a4_embed, a4_l)

        concat_a0 = torch.cat([ctx_embed, u_a0, u_q, u_a0 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a1 = torch.cat([ctx_embed, u_a1, u_q, u_a1 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a2 = torch.cat([ctx_embed, u_a2, u_q, u_a2 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a3 = torch.cat([ctx_embed, u_a3, u_q, u_a3 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a4 = torch.cat([ctx_embed, u_a4, u_q, u_a4 * ctx_embed, u_q * ctx_embed], dim=-1)

        concat_a0 = self.merge(concat_a0)
        concat_a1 = self.merge(concat_a1)
        concat_a2 = self.merge(concat_a2)
        concat_a3 = self.merge(concat_a3)
        concat_a4 = self.merge(concat_a4)



        mature_maxout_a0 = max_along_time(concat_a0, ctx_l).unsqueeze(1)
        mature_maxout_a1 = max_along_time(concat_a1, ctx_l).unsqueeze(1)
        mature_maxout_a2 = max_along_time(concat_a2, ctx_l).unsqueeze(1)
        mature_maxout_a3 = max_along_time(concat_a3, ctx_l).unsqueeze(1)
        mature_maxout_a4 = max_along_time(concat_a4, ctx_l).unsqueeze(1)

        mature_answers = torch.cat([
            mature_maxout_a0, mature_maxout_a1, mature_maxout_a2, mature_maxout_a3, mature_maxout_a4
        ], dim=1)
        out = classifier(mature_answers)  # (B, 5)
        return out

class CNN_simple(nn.Module):
    def __init__(self, opt):
        super(CNN_simple, self).__init__()
        self.vid_flag = "imagenet" in opt.input_streams
        self.sub_flag = "sub" in opt.input_streams
        self.vcpt_flag = "vcpt" in opt.input_streams
        # hidden_size_1 = opt.hsz1
        hidden_size = opt.hsz2
        n_layers_cls = opt.n_layers_cls
        vid_feat_size = opt.vid_feat_size
        embedding_size = opt.embedding_size
        vocab_size = opt.vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bidaf = BidafAttn(hidden_size * 3, method="dot")  # no parameter for dot
        self.merge = nn.Sequential(
            nn.Linear(embedding_size * 5, hidden_size),
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
            self.cnn_vid = nn.Conv1d(hidden_size, hidden_size, 7, padding=3)
            self.classifier_vid = MLP(hidden_size, 1, 500, n_layers_cls)

        if self.sub_flag:
            print("activate sub stream")
            self.cnn_sub = nn.Conv1d(hidden_size, hidden_size, 7, padding=3)
            self.classifier_sub = MLP(hidden_size, 1, 500, n_layers_cls)

        if self.vcpt_flag:
            print("activate vcpt stream")
            self.cnn_vcpt = nn.Conv1d(hidden_size, hidden_size, 7, padding=3)
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


        if self.sub_flag:
            e_sub = self.embedding(sub)
            sub_out = self.stream_processor(self.cnn_sub, self.classifier_sub, e_sub, sub_l,
                                            e_q, q_l, e_a0, a0_l, e_a1, a1_l,
                                            e_a2, a2_l, e_a3, a3_l, e_a4, a4_l)
        else:
            sub_out = 0

        if self.vcpt_flag:
            e_vcpt = self.embedding(vcpt)
            vcpt_out = self.stream_processor(self.cnn_vcpt, self.classifier_vcpt, e_vcpt, vcpt_l,
                                             e_q, q_l, e_a0, a0_l, e_a1, a1_l,
                                             e_a2, a2_l, e_a3, a3_l, e_a4, a4_l)
        else:
            vcpt_out = 0

        if self.vid_flag:
            e_vid = self.video_fc(vid)
            vid_out = self.stream_processor(self.cnn_vid, self.classifier_vid, e_vid, vid_l,
                                            e_q, q_l, e_a0, a0_l, e_a1, a1_l,
                                            e_a2, a2_l, e_a3, a3_l, e_a4, a4_l)
        else:
            vid_out = 0
        edge_prob_sub = 0
        edge_prob_vcpt = 0
        edge_prob_vid = 0
        out = sub_out + vcpt_out + vid_out  # adding zeros has no effect on backward
        return out.squeeze(), edge_prob_sub, edge_prob_vcpt, edge_prob_vid




    def stream_processor(self, seq_model, classifier, ctx_embed, ctx_l,
                         q_embed, q_l, a0_embed, a0_l, a1_embed, a1_l, a2_embed, a2_l, a3_embed, a3_l, a4_embed, a4_l):

        ctx_embed = seq_model(ctx_embed).transpose(1, 2)

        u_q, _ = self.bidaf(ctx_embed, ctx_l, q_embed, q_l)
        u_a0, _ = self.bidaf(ctx_embed, ctx_l, a0_embed, a0_l)
        u_a1, _ = self.bidaf(ctx_embed, ctx_l, a1_embed, a1_l)
        u_a2, _ = self.bidaf(ctx_embed, ctx_l, a2_embed, a2_l)
        u_a3, _ = self.bidaf(ctx_embed, ctx_l, a3_embed, a3_l)
        u_a4, _ = self.bidaf(ctx_embed, ctx_l, a4_embed, a4_l)

        concat_a0 = torch.cat([ctx_embed, u_a0, u_q, u_a0 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a1 = torch.cat([ctx_embed, u_a1, u_q, u_a1 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a2 = torch.cat([ctx_embed, u_a2, u_q, u_a2 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a3 = torch.cat([ctx_embed, u_a3, u_q, u_a3 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a4 = torch.cat([ctx_embed, u_a4, u_q, u_a4 * ctx_embed, u_q * ctx_embed], dim=-1)

        concat_a0 = self.merge(concat_a0)
        concat_a1 = self.merge(concat_a1)
        concat_a2 = self.merge(concat_a2)
        concat_a3 = self.merge(concat_a3)
        concat_a4 = self.merge(concat_a4)



        mature_maxout_a0 = max_along_time(concat_a0, ctx_l).unsqueeze(1)
        mature_maxout_a1 = max_along_time(concat_a1, ctx_l).unsqueeze(1)
        mature_maxout_a2 = max_along_time(concat_a2, ctx_l).unsqueeze(1)
        mature_maxout_a3 = max_along_time(concat_a3, ctx_l).unsqueeze(1)
        mature_maxout_a4 = max_along_time(concat_a4, ctx_l).unsqueeze(1)

        mature_answers = torch.cat([
            mature_maxout_a0, mature_maxout_a1, mature_maxout_a2, mature_maxout_a3, mature_maxout_a4
        ], dim=1)
        out = classifier(mature_answers)  # (B, 5)
        return out

class selfattention_simple(nn.Module):
    def __init__(self, opt):
        super(selfattention_simple, self).__init__()
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
        #self.position_encoding = PositionEncoding(n_filters=embedding_size)

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bidaf = BidafAttn(hidden_size * 3, method="dot")  # no parameter for dot

        self.merge = nn.Sequential(
            nn.Linear(embedding_size * 5, hidden_size),
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

        #e_q = self.position_encoding(e_q)
        #e_a0 = self.position_encoding(e_a0)
        #e_a1 = self.position_encoding(e_a1)
        #e_a2 = self.position_encoding(e_a2)
        #e_a3 = self.position_encoding(e_a3)
        #e_a4 = self.position_encoding(e_a4)


        #raw_out_q, _ = self.lstm_raw(e_q, q_l)
        #raw_out_a0, _ = self.lstm_raw(e_a0, a0_l)
        #raw_out_a1, _ = self.lstm_raw(e_a1, a1_l)
        #raw_out_a2, _ = self.lstm_raw(e_a2, a2_l)
        #raw_out_a3, _ = self.lstm_raw(e_a3, a3_l)
        #raw_out_a4, _ = self.lstm_raw(e_a4, a4_l)

        if self.sub_flag:
            e_sub = self.embedding(sub)
            #e_sub = self.position_encoding(e_sub)

            #raw_out_sub, _ = self.lstm_raw(e_sub, sub_l)
            sub_out = self.stream_processor(self.sequence_sub, self.classifier_sub, e_sub, sub_l,
                                            e_q, q_l, e_a0, a0_l, e_a1, a1_l,
                                            e_a2, a2_l, e_a3, a3_l, e_a4, a4_l)
        else:
            sub_out = 0

        if self.vcpt_flag:
            e_vcpt = self.embedding(vcpt)
            #e_vcpt = self.position_encoding(e_vcpt)

            #raw_out_vcpt, _ = self.lstm_raw(e_vcpt, vcpt_l)
            vcpt_out = self.stream_processor(self.sequence_vcpt, self.classifier_vcpt, e_vcpt, vcpt_l,
                                             e_q, q_l, e_a0, a0_l, e_a1, a1_l,
                                             e_a2, a2_l, e_a3, a3_l, e_a4, a4_l)
        else:
            vcpt_out = 0

        if self.vid_flag:
            e_vid = self.video_fc(vid)
            #e_vid = self.position_encoding(e_vid)

            #raw_out_vid, _ = self.lstm_raw(e_vid, vid_l)
            vid_out = self.stream_processor(self.sequence_vid, self.classifier_vid, e_vid, vid_l,
                                            e_q, q_l, e_a0, a0_l, e_a1, a1_l,
                                            e_a2, a2_l, e_a3, a3_l, e_a4, a4_l)
        else:
            vid_out = 0
        edge_prob_sub = 0
        edge_prob_vcpt = 0
        edge_prob_vid = 0
        out = sub_out + vcpt_out + vid_out  # adding zeros has no effect on backward
        return out.squeeze(), edge_prob_sub, edge_prob_vcpt, edge_prob_vid

    def stream_processor(self, sequence_model, classifier, ctx_embed, ctx_l,
                         q_embed, q_l, a0_embed, a0_l, a1_embed, a1_l, a2_embed, a2_l, a3_embed, a3_l, a4_embed, a4_l):

        ctx_embed, _ = sequence_model(ctx_embed, ctx_l)

        u_q, _ = self.bidaf(ctx_embed, ctx_l, q_embed, q_l)
        u_a0, _ = self.bidaf(ctx_embed, ctx_l, a0_embed, a0_l)
        u_a1, _ = self.bidaf(ctx_embed, ctx_l, a1_embed, a1_l)
        u_a2, _ = self.bidaf(ctx_embed, ctx_l, a2_embed, a2_l)
        u_a3, _ = self.bidaf(ctx_embed, ctx_l, a3_embed, a3_l)
        u_a4, _ = self.bidaf(ctx_embed, ctx_l, a4_embed, a4_l)

        concat_a0 = torch.cat([ctx_embed, u_a0, u_q, u_a0 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a1 = torch.cat([ctx_embed, u_a1, u_q, u_a1 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a2 = torch.cat([ctx_embed, u_a2, u_q, u_a2 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a3 = torch.cat([ctx_embed, u_a3, u_q, u_a3 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a4 = torch.cat([ctx_embed, u_a4, u_q, u_a4 * ctx_embed, u_q * ctx_embed], dim=-1)

        concat_a0 = self.merge(concat_a0)
        concat_a1 = self.merge(concat_a1)
        concat_a2 = self.merge(concat_a2)
        concat_a3 = self.merge(concat_a3)
        concat_a4 = self.merge(concat_a4)

        mature_maxout_a0 = max_along_time(concat_a0, ctx_l).unsqueeze(1)
        mature_maxout_a1 = max_along_time(concat_a1, ctx_l).unsqueeze(1)
        mature_maxout_a2 = max_along_time(concat_a2, ctx_l).unsqueeze(1)
        mature_maxout_a3 = max_along_time(concat_a3, ctx_l).unsqueeze(1)
        mature_maxout_a4 = max_along_time(concat_a4, ctx_l).unsqueeze(1)

        mature_answers = torch.cat([
            mature_maxout_a0, mature_maxout_a1, mature_maxout_a2, mature_maxout_a3, mature_maxout_a4
        ], dim=1)
        out = classifier(mature_answers)  # (B, 5)
        return out


class sparse_selfattention_simple(nn.Module):
    def __init__(self, opt):
        super(sparse_selfattention_simple, self).__init__()
        self.vid_flag = "imagenet" in opt.input_streams
        self.sub_flag = "sub" in opt.input_streams
        self.vcpt_flag = "vcpt" in opt.input_streams
        self.prior = opt.prior
        #hidden_size_1 = opt.hsz1
        hidden_size = opt.hsz2
        n_layers_cls = opt.n_layers_cls
        vid_feat_size = opt.vid_feat_size
        embedding_size = opt.embedding_size
        vocab_size = opt.vocab_size
        head_num = opt.multihead
        prior = opt.prior
        #self.position_encoding = PositionEncoding(n_filters=embedding_size)

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bidaf = BidafAttn(hidden_size * 3, method="dot")  # no parameter for dot

        self.merge = nn.Sequential(
            nn.Linear(embedding_size * 5, hidden_size),
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
            self.sequence_vid = SparseSelfAttention(hidden_size, hidden_size, dropout_p=0.,
                                              attention_head=head_num, prior = prior)
            self.classifier_vid = MLP(hidden_size, 1, 500, n_layers_cls)

        if self.sub_flag:
            print("activate sub stream")
            self.sequence_sub = SparseSelfAttention(hidden_size, hidden_size, dropout_p=0.,
                                              attention_head=head_num, prior = prior)
            self.classifier_sub = MLP(hidden_size, 1, 500, n_layers_cls)

        if self.vcpt_flag:
            print("activate vcpt stream")
            self.sequence_vcpt = SparseSelfAttention(hidden_size, hidden_size, dropout_p=0.,
                                              attention_head=head_num, prior = prior)
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


        if self.sub_flag:
            e_sub = self.embedding(sub)
            #e_sub = self.position_encoding(e_sub)

            #raw_out_sub, _ = self.lstm_raw(e_sub, sub_l)
            sub_out, edge_prob_sub = self.stream_processor(self.sequence_sub, self.classifier_sub, e_sub, sub_l,
                                            e_q, q_l, e_a0, a0_l, e_a1, a1_l,
                                            e_a2, a2_l, e_a3, a3_l, e_a4, a4_l)
        else:
            sub_out = 0
            edge_prob_sub = 0

        if self.vcpt_flag:
            e_vcpt = self.embedding(vcpt)
            #e_vcpt = self.position_encoding(e_vcpt)

            #raw_out_vcpt, _ = self.lstm_raw(e_vcpt, vcpt_l)
            vcpt_out, edge_prob_vcpt  = self.stream_processor(self.sequence_vcpt, self.classifier_vcpt, e_vcpt, vcpt_l,
                                             e_q, q_l, e_a0, a0_l, e_a1, a1_l,
                                             e_a2, a2_l, e_a3, a3_l, e_a4, a4_l)
        else:
            vcpt_out = 0
            edge_prob_vcpt = 0
        if self.vid_flag:
            e_vid = self.video_fc(vid)
            #e_vid = self.position_encoding(e_vid)

            #raw_out_vid, _ = self.lstm_raw(e_vid, vid_l)
            vid_out, edge_prob_vid = self.stream_processor(self.sequence_vid, self.classifier_vid, e_vid, vid_l,
                                            e_q, q_l, e_a0, a0_l, e_a1, a1_l,
                                            e_a2, a2_l, e_a3, a3_l, e_a4, a4_l)
        else:
            vid_out = 0
            edge_prob_vid = 0
        out = sub_out + vcpt_out + vid_out  # adding zeros has no effect on backward
        return out.squeeze(), edge_prob_sub, edge_prob_vcpt, edge_prob_vid

    def stream_processor(self, sequence_model, classifier, ctx_embed, ctx_l,
                         q_embed, q_l, a0_embed, a0_l, a1_embed, a1_l, a2_embed, a2_l, a3_embed, a3_l, a4_embed, a4_l):
        ctx_embed, _, edge_prob = sequence_model(ctx_embed, ctx_l)

        u_q, _ = self.bidaf(ctx_embed, ctx_l, q_embed, q_l)
        u_a0, _ = self.bidaf(ctx_embed, ctx_l, a0_embed, a0_l)
        u_a1, _ = self.bidaf(ctx_embed, ctx_l, a1_embed, a1_l)
        u_a2, _ = self.bidaf(ctx_embed, ctx_l, a2_embed, a2_l)
        u_a3, _ = self.bidaf(ctx_embed, ctx_l, a3_embed, a3_l)
        u_a4, _ = self.bidaf(ctx_embed, ctx_l, a4_embed, a4_l)

        concat_a0 = torch.cat([ctx_embed, u_a0, u_q, u_a0 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a1 = torch.cat([ctx_embed, u_a1, u_q, u_a1 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a2 = torch.cat([ctx_embed, u_a2, u_q, u_a2 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a3 = torch.cat([ctx_embed, u_a3, u_q, u_a3 * ctx_embed, u_q * ctx_embed], dim=-1)
        concat_a4 = torch.cat([ctx_embed, u_a4, u_q, u_a4 * ctx_embed, u_q * ctx_embed], dim=-1)

        concat_a0 = self.merge(concat_a0)
        concat_a1 = self.merge(concat_a1)
        concat_a2 = self.merge(concat_a2)
        concat_a3 = self.merge(concat_a3)
        concat_a4 = self.merge(concat_a4)

        #concat_a0 = self.position_encoding(concat_a0)
        #concat_a1 = self.position_encoding(concat_a1)
        #concat_a2 = self.position_encoding(concat_a2)
        #concat_a3 = self.position_encoding(concat_a3)
        #concat_a4 = self.position_encoding(concat_a4)



        mature_maxout_a0 = max_along_time(concat_a0, ctx_l).unsqueeze(1)
        mature_maxout_a1 = max_along_time(concat_a1, ctx_l).unsqueeze(1)
        mature_maxout_a2 = max_along_time(concat_a2, ctx_l).unsqueeze(1)
        mature_maxout_a3 = max_along_time(concat_a3, ctx_l).unsqueeze(1)
        mature_maxout_a4 = max_along_time(concat_a4, ctx_l).unsqueeze(1)

        mature_answers = torch.cat([
            mature_maxout_a0, mature_maxout_a1, mature_maxout_a2, mature_maxout_a3, mature_maxout_a4
        ], dim=1)
        out = classifier(mature_answers)  # (B, 5)
        return out, edge_prob



