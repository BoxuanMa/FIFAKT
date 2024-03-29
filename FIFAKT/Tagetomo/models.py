import torch
from torch import nn
import math
from torch.nn import functional as F
from enum import IntEnum
import numpy as np
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBUG = True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes, device=device)[y]


def loss_with_z_term(loss_fn, z_hat, y, class_weights=None, seen=None, z_weight=1.0, eps=1e-6):
    y_clamp = torch.clamp(y, eps, 1.0 - eps)
    # range (min,max) 限幅
    z = torch.log(y_clamp / (1-y_clamp))

    if seen is not None:
        return (loss_fn(z_hat, y).flatten() + z_weight * torch.square(z - z_hat).flatten()) * seen
    else:
        if class_weights is not None:
            weight_indices = torch.floor(y / 0.1).long().view(-1)
            weight_indices[weight_indices == 10] = 9
            class_weights_to_apply = class_weights[weight_indices]
            loss_intermediate = loss_fn(z_hat, y).view(-1) + z_weight * torch.square(z - z_hat).view(-1)
            return loss_intermediate * class_weights_to_apply
        return loss_fn(z_hat, y) + z_weight * torch.square(z - z_hat)



class DKT(nn.Module):
    def __init__(self, n_question, embed_l, hidden_dim, layer_dim, output_dim, class_weights=None, z_weight=0.0):
        super(DKT, self).__init__()
        self.n_question = n_question
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.z_weight = z_weight
        self.class_weights = class_weights
        embed=64
        self.rnn = nn.LSTM(embed, hidden_dim, layer_dim)

        self.out = torch.nn.Linear(self.hidden_dim, self.n_question)

        self.sig = nn.Sigmoid()
        self.q_embed = nn.Embedding(self.n_question*2, embed)


    def forward(self, q_data, format_data,  target, mask):
        # q_embed_data = self.q_embed(q_data)  # seqlen, BS,   d_model
        target_l=target.long()
        q_embed_data = self.q_embed(q_data + abs(target_l) * self.n_question)


        batch_size, sl = q_data.size(1), q_data.size(0)
        h_0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=device)

        rnn_input = torch.cat([q_embed_data],dim=2)

        out, (hn,cn) = self.rnn(rnn_input, (h_0,c_0))

        h = torch.cat([h_0[-1:, :, :], out], dim=0)[:-1, :, :]

        target_item = q_data[:,1:]
        pred = self.out(out)
        preds = torch.gather(pred, dim=-1, index=target_item.unsqueeze(dim=-1)).squeeze(dim=-1)
        labels = target[:, 1:].contiguous().view(-1)
        m = nn.Sigmoid()
        preds = preds.view(-1)  # logit

        mask=mask[:, 1:].contiguous().view(-1)

        masked_labels = labels[mask]
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        out = loss_with_z_term(loss, masked_preds, masked_labels, class_weights=self.class_weights, z_weight=self.z_weight)
        return out, m(preds), mask.sum()


####
class FIFAKT(nn.Module):
    def __init__(self, n_question, embed_l, hidden_dim, layer_dim=1, class_weights=None, final_fc_dim=512, dropout=0.0, z_weight=0.0, pretrained_embeddings=None, freeze_pretrained=True):
        super(FIFAKT, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.z_weight = z_weight
        self.class_weights = class_weights
        self.n_question = n_question
        self.layer_dim = layer_dim

        # pretrained_embeddings=None

        if pretrained_embeddings is not None:
            print("embeddings frozen:", freeze_pretrained, flush=True)
            self.q_embed = nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=0,freeze=freeze_pretrained)
        else:
            self.q_embed = nn.Embedding(self.n_question, embed_l, padding_idx=0)
        # self.q_embed = nn.Embedding(self.n_question, embed_l, padding_idx=0)
        num_features = 6
        format_emb = 13
        self.rnn = nn.LSTM(
            input_size=embed_l + 1 + num_features+format_emb,
            hidden_size=self.hidden_dim,
            num_layers=self.layer_dim,
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_dim*2 + embed_l+format_emb + num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )


    def attention_net_q(self, q_context, state, l):
        # hidden = final_state.view(-1, self.hidden_dim , 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        q_context_t= q_context.transpose(1, 2)

        attn_weights_o = torch.bmm(q_context, q_context_t).squeeze(2) # attn_weights : [batch_size, n_step]

        attn_weights = attn_weights_o[:, :, 1:]
        scaled_attn_weights = torch.divide(attn_weights, math.sqrt(l))
        scaled_attn_weights = torch.triu(scaled_attn_weights)
        scaled_attn_weights[scaled_attn_weights == 0] = -1000

        soft_attn_weights = F.softmax(scaled_attn_weights, dim=-2)
        soft_attn_weights= torch.triu(soft_attn_weights)
        context = torch.bmm(state.transpose(1, 2), soft_attn_weights).squeeze(2)

        context = context.transpose(1, 2)


        return context, soft_attn_weights.data

    def forward(self, q_data, format_data, wordsize_data, correct_data, attempts_data, delta, delta_t, delta_repeat, time_bin_data, time_bin_t_data, target, mask):
        q_embed_data = self.q_embed(q_data) # input : [batch_size, len_seq, embedding_dim]

        format_onehot = to_categorical(format_data, 13)

        batch_size, sl = q_data.size(1), q_data.size(0)

        hidden_state = Variable(torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=device)) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=device)) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]


        rnn_input = torch.cat([q_embed_data, format_onehot, torch.unsqueeze(target, 2), torch.unsqueeze(wordsize_data, 2),torch.unsqueeze(correct_data, 2), torch.unsqueeze(attempts_data, 2),torch.unsqueeze(delta_repeat, 2), torch.unsqueeze(delta, 2),torch.unsqueeze(delta_t, 2)], dim=2)
        # rnn_input = torch.cat([q_embed_data,format_onehot, torch.unsqueeze(target, 2)], dim=2)

        output, (final_hidden_state, final_cell_state) = self.rnn(rnn_input, (hidden_state, cell_state))




        att_input = torch.cat([q_embed_data, format_onehot], dim=2)
        attn_output, attention = self.attention_net_q(att_input, output, l=len(att_input))
        ffn_input = torch.cat([attn_output, output[:,:-1,:],  q_embed_data[:, 1:], format_onehot[:, 1:], torch.unsqueeze(wordsize_data[:, 1:], 2), torch.unsqueeze(correct_data[:, 1:], 2),torch.unsqueeze(attempts_data[:, 1:], 2), torch.unsqueeze(delta_repeat[:, 1:], 2), torch.unsqueeze(delta[:, 1:], 2),torch.unsqueeze(delta_t[:, 1:], 2)], dim=2)

        pred = self.out(ffn_input)
        labels = target[:, 1:].contiguous().view(-1)


        m = nn.Sigmoid()
        preds = pred.view(-1)  # logit
        mask = mask[:, 1:].contiguous().view(-1)
        masked_labels = labels[mask]
        masked_preds = preds[mask]


        loss = nn.BCEWithLogitsLoss(reduction='none')
        out = loss_with_z_term(loss, masked_preds, masked_labels, class_weights=self.class_weights, z_weight=self.z_weight)
        return out, m(preds), mask.sum(),attention
        # return , attention # model : [batch_size, num_classes], attention : [batch_size, n_step]
