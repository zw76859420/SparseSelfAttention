__author__ = "Jie Lei"

#  ref: https://github.com/lichengunc/MAttNet/blob/master/lib/layers/lang_encoder.py#L11
#  ref: https://github.com/easonnie/flint/blob/master/torch_util.py#L272
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.autograd import Variable


class MultiHeadAttention(nn.Module):
    """A RNN wrapper handles variable length inputs, always set batch_first=True.
    Supports LSTM, GRU and RNN. Tested with PyTorch 0.3 and 0.4
    """

    def __init__(self, input_size, hidden_size, dropout_p=0, attention_head=1):
        super(SelfAttention, self).__init__()

        """  
	    :param word_embedding_size: rnn input size
	    :param hidden_size: rnn output size
	    :param dropout_p: between rnn layers, only useful when n_layer >= 2
	    """
        self.hidden_dim = hidden_size
        self.attention_head = attention_head
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)
        self.key_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Dropout(dropout_p),
        )
        self.query_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Dropout(dropout_p),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Dropout(dropout_p),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        self.LN1 = nn.LayerNorm([input_size])
        self.LN2 = nn.LayerNorm([input_size])

    def forward(self, queries, inputs, lengths):
        """
        inputs, sorted_inputs -> (B, T, D)
        lengths -> (B, )
        outputs -> (B, T, n_dirs * D)
        hidden -> (n_layers * n_dirs, B, D) -> (B, n_dirs * D)  keep the last layer
        - add total_length in pad_packed_sequence for compatiblity with nn.DataParallel, --remove it
        """
        assert len(inputs) == len(lengths)
        dim = inputs.size(-1)
        input_mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths)).cuda() >= lengths.unsqueeze(
            1)  # [B,T]

        input_mask_ = input_mask.unsqueeze(2)
        inputs.data.masked_fill_(input_mask_.data.bool(), 0.)

        k = self.key_layer(inputs)
        q = self.query_layer(queries)
        v = self.value_layer(inputs)

        k = split_last_dimension(k, self.attention_head)  # [B,T,h,T//h]
        k = k.permute(0, 2, 1, 3)  # [B,h,T,T//h]
        q = split_last_dimension(q, self.attention_head)  # [B,T,h,T//h]
        q = q.permute(0, 2, 1, 3)  # [B,h,T,T//h]
        v = split_last_dimension(v, self.attention_head)  # [B,T,h,T//h]
        v = v.permute(0, 2, 1, 3)  # [B,h,T,T//h]

        qk = torch.matmul(q, k.transpose(2, 3))  # [B, h, T, T]
        qk /= dim ** 0.5

        input_mask_ = input_mask.unsqueeze(1)
        input_mask_ = input_mask_.unsqueeze(1)  # [B,1,1,T]
        qk.masked_fill_(input_mask_.data.bool(), -float("inf"))
        qk = self.softmax(qk)

        qkv = torch.matmul(qk, v)  # [B, h, T, H/h]
        qkv = F.relu(qkv)

        qkv = qkv.permute(0, 2, 1, 3)
        # print(qkv.size())
        qkv = qkv.contiguous()
        qkv = qkv.view(qkv.size(0), qkv.size(1), -1)

        midl_output = self.dropout(qkv)
        midl_output_ = midl_output + inputs

        midl_output_ = self.LN1(midl_output_)

        midl_output = self.fully_connected(midl_output_)
        midl_output = midl_output + midl_output_
        outputs = self.LN2(midl_output)

        input_mask_ = input_mask.unsqueeze(2)
        outputs.masked_fill_(input_mask_.data.bool(), 0.)

        return outputs, qk

    
    
    
    
    
    
class SelfAttention(nn.Module):
    """A RNN wrapper handles variable length inputs, always set batch_first=True.
    Supports LSTM, GRU and RNN. Tested with PyTorch 0.3 and 0.4
    """

    def __init__(self, input_size, hidden_size, dropout_p=0, attention_head=1):
        super(SelfAttention, self).__init__()

        """  
	    :param word_embedding_size: rnn input size
	    :param hidden_size: rnn output size
	    :param dropout_p: between rnn layers, only useful when n_layer >= 2
	    """
        self.hidden_dim = hidden_size
        self.attention_head = attention_head
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)
        self.key_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Dropout(dropout_p),
        )
        self.query_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Dropout(dropout_p),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Dropout(dropout_p),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        self.LN1 = nn.LayerNorm([input_size])
        self.LN2 = nn.LayerNorm([input_size])

    def forward(self, inputs, lengths):
        """
        inputs, sorted_inputs -> (B, T, D)
        lengths -> (B, )
        outputs -> (B, T, n_dirs * D)
        hidden -> (n_layers * n_dirs, B, D) -> (B, n_dirs * D)  keep the last layer
        - add total_length in pad_packed_sequence for compatiblity with nn.DataParallel, --remove it
        """
        assert len(inputs) == len(lengths)
        dim = inputs.size(-1)
#         input_mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths)).cuda() >= lengths.unsqueeze(
#             1)  # [B,T]
        input_mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths)) >= lengths.unsqueeze(
            1)  # [B,T]

        input_mask_ = input_mask.unsqueeze(2)
        inputs.data.masked_fill_(input_mask_.data.bool(), 0.)

        k = self.key_layer(inputs)
        q = self.query_layer(inputs)
        v = self.value_layer(inputs)

        k = split_last_dimension(k, self.attention_head)  # [B,T,h,T//h]
        k = k.permute(0, 2, 1, 3)  # [B,h,T,T//h]
        q = split_last_dimension(q, self.attention_head)  # [B,T,h,T//h]
        q = q.permute(0, 2, 1, 3)  # [B,h,T,T//h]
        v = split_last_dimension(v, self.attention_head)  # [B,T,h,T//h]
        v = v.permute(0, 2, 1, 3)  # [B,h,T,T//h]

        qk = torch.matmul(q, k.transpose(2, 3))  # [B, h, T, T]
        qk /= dim ** 0.5

        input_mask_ = input_mask.unsqueeze(1)
        input_mask_ = input_mask_.unsqueeze(1)  # [B,1,1,T]
        qk.masked_fill_(input_mask_.data.bool(), -float("inf"))
        qk = self.softmax(qk)

        qkv = torch.matmul(qk, v)  # [B, h, T, H/h]
        qkv = F.relu(qkv)

        qkv = qkv.permute(0, 2, 1, 3)
        # print(qkv.size())
        qkv = qkv.contiguous()
        qkv = qkv.view(qkv.size(0), qkv.size(1), -1)

        midl_output = self.dropout(qkv)
        midl_output_ = midl_output + inputs

        midl_output_ = self.LN1(midl_output_)

        midl_output = self.fully_connected(midl_output_)
        midl_output = midl_output + midl_output_
        outputs = self.LN2(midl_output)

        input_mask_ = input_mask.unsqueeze(2)
        outputs.masked_fill_(input_mask_.data.bool(), 0.)

        return outputs, qk

    
    
    
    

class SparseSelfAttention(nn.Module):
    """A RNN wrapper handles variable length inputs, always set batch_first=True.
    Supports LSTM, GRU and RNN. Tested with PyTorch 0.3 and 0.4
    """

    def __init__(self, input_size, hidden_size, dropout_p=0, attention_head=1, prior="None"):
        super(SparseSelfAttention, self).__init__()

        """  
	    :param word_embedding_size: rnn input size
	    :param hidden_size: rnn output size
	    :param dropout_p: between rnn layers, only useful when n_layer >= 2
	    """
        self.hidden_dim = hidden_size
        self.attention_head = attention_head
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)
        self.prior = prior

        self.key_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Dropout(dropout_p),
        )
        self.query_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Dropout(dropout_p),
        )

        self.value_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Dropout(dropout_p),
        )

        self.edgek_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Dropout(dropout_p),
        )
        self.edgeq_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Dropout(dropout_p),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        self.LN1 = nn.LayerNorm([input_size])
        self.LN2 = nn.LayerNorm([input_size])

    def forward(self, inputs, lengths):
        """
        inputs, sorted_inputs -> (B, T, D)
        lengths -> (B, )
        outputs -> (B, T, n_dirs * D)
        hidden -> (n_layers * n_dirs, B, D) -> (B, n_dirs * D)  keep the last layer
        - add total_length in pad_packed_sequence for compatiblity with nn.DataParallel, --remove it
        """
        assert len(inputs) == len(lengths)
        dim = inputs.size(-1)
        input_mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths)).cuda() >= lengths.unsqueeze(
            1)  # [B,T]

        input_mask_ = input_mask.unsqueeze(2)
        inputs.data.masked_fill_(input_mask_.data.bool(), 0.)

        k = self.key_layer(inputs)
        q = self.query_layer(inputs)
        v = self.value_layer(inputs)


        k = split_last_dimension(k, self.attention_head)  # [B,T,h,T//h]
        k = k.permute(0, 2, 1, 3)  # [B,h,T,T//h]
        q = split_last_dimension(q, self.attention_head)  # [B,T,h,T//h]
        q = q.permute(0, 2, 1, 3)  # [B,h,T,T//h]
        v = split_last_dimension(v, self.attention_head)  # [B,T,h,T//h]
        v = v.permute(0, 2, 1, 3)  # [B,h,T,T//h]

        qk = torch.matmul(q, k.transpose(2, 3))  # [B, h, T, T]
        qk /= dim ** 0.5  # TODO: exist or not check

        if 1:
            edgek = self.edgek_layer(inputs)
            edgeq = self.edgeq_layer(inputs)

            edgek = split_last_dimension(edgek, self.attention_head)  # [B,T,h,T//h]
            edgek = edgek.permute(0, 2, 1, 3)  # [B,h,T,T//h]
            edgeq = split_last_dimension(edgeq, self.attention_head)  # [B,T,h,T//h]
            edgeq = edgeq.permute(0, 2, 1, 3)
            edgeqk = torch.matmul(edgeq, edgek.transpose(2, 3))
        else:
            edgeqk = qk.contiguous()

        if self.prior == "categorical":
            edges = gumbel_softmax(edgeqk, hard=True)

            edges_prob = pr_softmax(qk, 1)
            off_edge = torch.ones_like(qk)*-float("inf")
            qk = torch.where(edges.data.bool(), qk, off_edge)
        elif self.prior == "bernoulli":
            #print("HERE")
            edges_prob = torch.sigmoid(edgeqk)
            #print(edges_prob)
            edges = binary_concrete(edges_prob, hard=True)


            off_edge = torch.ones_like(qk) *-float("inf")
            qk = torch.where(edges.data.bool(), qk, off_edge)
        elif self.prior == "bernoulli_1":
            edgek = self.edgek_layer(inputs)
            edgeq = self.edgeq_layer(inputs)


            edgeqk = torch.matmul(edgeq, edgek.transpose(1, 2))
            edges_prob = torch.sigmoid(edgeqk)
            #print(edges_prob)
            #print(edges_prob.size())
            edges = binary_concrete(edges_prob, hard=True)

            off_edge = (torch.ones_like(qk) * -float("inf"))
            qk = torch.where(edges.data.bool().unsqueeze(1).expand_as(qk), qk, off_edge)




        input_mask_ = input_mask.unsqueeze(1)
        input_mask_ = input_mask_.unsqueeze(1)  # [B,1,1,T]
        qk.masked_fill_(input_mask_.data.bool(), -float("inf"))
        #qk = qk*edges
        #print("before softmax:", torch.isnan(qk).sum())
        qk = self.softmax(qk)
        #print("after softmax:", torch.isnan(qk).sum())
        #print(qk[4])


        qk = torch.where(torch.isnan(qk), torch.zeros_like(qk), qk)

        qkv = torch.matmul(qk, v)  # [B, h, T, H/h]
        qkv = F.relu(qkv)

        qkv = qkv.permute(0, 2, 1, 3)
        # print(qkv.size())
        qkv = qkv.contiguous()
        qkv = qkv.view(qkv.size(0), qkv.size(1), -1)

        midl_output = self.dropout(qkv)
        midl_output_ = midl_output + inputs

        midl_output_ = self.LN1(midl_output_)

        midl_output = self.fully_connected(midl_output_)
        midl_output = midl_output + midl_output_
        outputs = self.LN2(midl_output)

        input_mask_ = input_mask.unsqueeze(2)
        outputs.masked_fill_(input_mask_.data.bool(), 0.)

        return outputs, qk, edges_prob


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
      x: a Tensor with shape [..., m]
      n: an integer.
    Returns:
      a Tensor with shape [..., n, m/n]
    """
    x_shape = x.size()
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return x.view(x_shape[0], x_shape[1], n, m // n)

def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, axis, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return pr_softmax(y / tau, axis=axis)


def gumbel_softmax(logits, axis=-1, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    logits = logits.permute(0, 2, 3, 1).contiguous()  # multihead to last dim
    y_soft = gumbel_softmax_sample(logits, axis, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(axis)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y.permute(0, 3, 1, 2)


def pr_softmax(input, axis=-1):
    my_softmax = nn.Softmax(dim=axis)
    con_input = input.contiguous()
    output = my_softmax(con_input)
    return output


def binary_concrete(logits, tau=1, hard=False, eps=1e-10):
    y_soft = binary_concrete_sample(logits, tau=tau, eps=eps)
    if hard:
        #y = (y_soft > 0.5).data
        y_hard = (y_soft > 0.5).float()
        y = Variable(y_hard.data - y_soft.data) + y_soft

    else:
        y = y_soft
    return y


def binary_concrete_sample(logits, tau=1, eps=1e-10):
    logistic_noise = sample_logistic(logits.size(), eps=eps)
    if logits.is_cuda:
        logistic_noise = logistic_noise.cuda()
    y = torch.log(logits+eps) - torch.log(1-logits+eps) + Variable(logistic_noise)
    return torch.sigmoid(y / tau)


def sample_logistic(shape, eps=1e-10):
    uniform = torch.rand(shape).float()
    return torch.log(uniform + eps) - torch.log(1 - uniform + eps)