import torch
from torch import nn
import math
import torch.nn.functional as F
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
    def __init__(self, n_question, embed_l, n_time_bins, hidden_dim, num_layers=1, class_weights=None, final_fc_dim=512, dropout=0.0, z_weight=0.0, pretrained_embeddings=None, freeze_pretrained=True):
        super().__init__()
        """
        Input:
            n_question : number of concepts + 1. question = 0 is used for padding.
            n_time_bins : number of time bins + 1. bin = 0 is used for padding.
        """
        self.n_question = n_question
        self.n_time_bins = n_time_bins
        self.dropout = dropout
        self.z_weight = z_weight
        self.class_weights = class_weights

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            input_size=embed_l + self.n_time_bins + 2,  # word embedding, time bin, session num correct, session num attempted
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim + embed_l + self.n_time_bins, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 1)
        )

        if pretrained_embeddings is not None:
            print("embeddings frozen:", freeze_pretrained, flush=True)
            self.q_embed = nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=0, freeze=freeze_pretrained)
        else:
            self.q_embed = nn.Embedding(self.n_question, embed_l, padding_idx=0)

    def forward(self, q_data, correct_data, attempts_data, time_bin_data, target, mask):
        """
            input:
            q_data : shape seqlen,  batchsize, concept id, from 1 to NumConcept, 0 is padding
            qa_data : shape seqlen, batchsize, concept response id, from 1 to 2*NumConcept, 0 is padding
            target : shape seqlen, batchsize, -1 is for padding timesteps.
        """
        q_embed_data = self.q_embed(q_data)  # seqlen, BS,   d_model
        time_bin_categorical = to_categorical(time_bin_data, self.n_time_bins)

        batch_size, sl = q_data.size(1), q_data.size(0)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

        rnn_input = torch.cat([q_embed_data, time_bin_categorical, torch.unsqueeze(correct_data, 2), torch.unsqueeze(attempts_data, 2)], dim=2)
        hidden_seq, _ = self.rnn(rnn_input, h_0)
        h = torch.cat([h_0[-1:, :, :], hidden_seq], dim=0)[:-1, :, :]  # T,BS,hidden_dim
        ffn_input = torch.cat([h, q_embed_data, time_bin_categorical], dim=2)  # concatenate time-shifted hidden states with current question info

        pred = self.out(ffn_input)  # Size (Seqlen, BS, n_question+1)

        labels = target.view(-1)
        m = nn.Sigmoid()
        preds = pred.view(-1)  # logit

        # mask = labels != -1
        mask = mask.view(-1)
        masked_labels = labels[mask]
        masked_preds = preds[mask]

        loss = nn.BCEWithLogitsLoss(reduction='none')
        out = loss_with_z_term(loss, masked_preds, masked_labels, class_weights=self.class_weights, z_weight=self.z_weight)
        return out, m(preds), mask.sum()

class DKT_ini(nn.Module):
    def __init__(self, n_question, embed_l, hidden_dim, layer_dim, output_dim, class_weights=None, z_weight=0.0):
        super(DKT_ini, self).__init__()
        self.n_question = n_question
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.z_weight = z_weight
        self.class_weights = class_weights

        self.rnn = nn.RNN(embed_l+1, hidden_dim, layer_dim,  nonlinearity='tanh')
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()
        self.q_embed = nn.Embedding(self.n_question, embed_l, padding_idx=0)

    def forward(self, q_data, correct_data, target, mask):
        q_embed_data = self.q_embed(q_data)  # seqlen, BS,   d_model
        batch_size, sl = q_data.size(1), q_data.size(0)
        h_0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim, device=device)
        rnn_input = torch.cat([q_embed_data, torch.unsqueeze(correct_data, 2)],dim=2)

        out, hn = self.rnn(rnn_input, h_0)
        pred = self.sig(self.fc(out))

        labels = target.view(-1)
        m = nn.Sigmoid()
        preds = pred.view(-1)  # logit

        # mask = labels != -1
        mask = mask.view(-1)
        masked_labels = labels[mask]
        masked_preds = preds[mask]

        loss = nn.BCEWithLogitsLoss(reduction='none')
        out = loss_with_z_term(loss, masked_preds, masked_labels, class_weights=self.class_weights, z_weight=self.z_weight)
        return out, m(preds), mask.sum()
