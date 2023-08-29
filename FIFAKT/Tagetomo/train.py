import numpy as np
import argparse

import pandas as pd

from models import DKT,FIFAKT
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, mean_absolute_error
import os


random_seed=2019
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
scaler = torch.cuda.amp.GradScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device being used:", device, flush=True)

parser = argparse.ArgumentParser(description='Run DKT.')
parser.add_argument('--data', type=str, nargs='?', default='d2020')
parser.add_argument('--method', type=str, nargs='?', default='FIFAKT') #dkt ,dkt_f, dkt_plus,dkt_wordsize
parser.add_argument('--embeddings_file', type=str, nargs='?', default="E:\dataset_memory\d2020\en_word_embeddings_fastword.npy")
# parser.add_argument('--embeddings_file', type=str, nargs='?', default=None)
parser.add_argument('--out_filename', type=str, nargs='?', default='dkt_model.pt')
parser.add_argument('--freeze_embeddings', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--grid_search', type=bool, nargs='?', const=True, default=True)
options = parser.parse_args()

print("data=", options.data)
print("method=", options.method)

print("grid search:", options.grid_search, flush=True)


if options.data == 'd2020':
    os.chdir('E:\dataset_memory\d2020_longseq')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DuoDataset(torch.utils.data.Dataset):
    def __init__(self, skill_path, format_path, wordsize_path, correct_path, attempts_path, delta_path, delta_t_path, delta_repeat_path, time_bin_path, time_bin_t_path,target_path, mask_path):
        self.skills = np.load(skill_path).T
        self.correct = np.load(correct_path).T
        self.attempts = np.load(attempts_path).T
        self.format = np.load(format_path).T
        self.wordsize = np.load(wordsize_path).T
        # self.c_correct = np.load(c_correct_path).T
        # self.c_attempts = np.load(c_attempts_path).T

        self.delta = np.load(delta_path).T
        self.delta_t = np.load(delta_t_path).T
        self.delta_repeat = np.load(delta_repeat_path).T
        self.time_bins = np.load(time_bin_path).T
        self.time_bins_t = np.load(time_bin_t_path).T
        self.targets = np.load(target_path).T
        self.mask = np.load(mask_path).T

    def __getitem__(self, idx):
        item = {"skills": torch.LongTensor(self.skills[:, idx]),
                "correct": torch.FloatTensor(self.correct[:, idx]),
                "attempts": torch.FloatTensor(self.attempts[:, idx]),
                "format": torch.LongTensor(self.format[:, idx]),
                "wordsize": torch.FloatTensor(self.wordsize[:, idx]),
                # "c_correct": torch.FloatTensor(self.c_correct[:, idx]),
                # "c_attempts": torch.FloatTensor(self.c_attempts[:, idx]),

                "delta": torch.LongTensor(self.delta[:, idx]),
                "delta_t": torch.LongTensor(self.delta_t[:, idx]),
                "delta_repeat": torch.LongTensor(self.delta_repeat[:, idx]),
                "time_bins": torch.LongTensor(self.time_bins[:, idx]),
                "time_bins_t": torch.LongTensor(self.time_bins_t[:, idx]),
                "targets": torch.FloatTensor(self.targets[:, idx]),
                "mask": torch.BoolTensor(self.mask[:, idx])}
        return item

    def __len__(self):
        return np.shape(self.targets)[1]

    def real_size(self):
        return np.sum(self.mask)


# train_dataset = DuoDataset("train_skills.npy", "train_correct.npy", "train_seen.npy", "train_time_bin.npy", "train_p.npy", "train_eval_mask.npy")
# dev_dataset = DuoDataset("dev_skills.npy", "dev_correct.npy", "dev_seen.npy", "dev_time_bin.npy", "dev_p.npy", "dev_eval_mask.npy")
# test_dataset = DuoDataset("test_skills.npy", "test_correct.npy", "test_seen.npy", "test_time_bin.npy", "test_p.npy", "test_eval_mask.npy")

train_dataset = DuoDataset("train_skillsen.npy","train_formaten.npy","train_wordsizeen.npy", "train_correcten.npy", "train_seenen.npy", "train_deltaen.npy", "train_delta_ten.npy","train_delta_repeaten.npy", "train_time_binen.npy","train_time_bin_ten.npy", "train_pen.npy", "train_eval_masken.npy")
dev_dataset = DuoDataset("dev_skillsen.npy", "dev_formaten.npy","dev_wordsizeen.npy","dev_correcten.npy", "dev_seenen.npy", "dev_deltaen.npy", "dev_delta_ten.npy", "dev_delta_repeaten.npy", "dev_time_binen.npy", "dev_time_bin_ten.npy","dev_pen.npy", "dev_eval_masken.npy")
test_dataset = DuoDataset("test_skillsen.npy","test_formaten.npy", "test_wordsizeen.npy", "test_correcten.npy", "test_seenen.npy",  "test_deltaen.npy", "test_delta_ten.npy","test_delta_repeaten.npy", "test_time_binen.npy","test_time_bin_ten.npy", "test_pen.npy", "test_eval_masken.npy")



if options.embeddings_file is not None:
    print("Using pretrained embeddings.", flush=True)
    embeddings = torch.FloatTensor(np.load(options.embeddings_file))
    print("embeddings shape:", np.shape(embeddings), flush=True)
    n_question, embed_dim = np.shape(embeddings)
    print(embeddings)
else:
    embeddings = None
    print("Training embeddings from scratch.", flush=True)
    n_question, embed_dim = 1901, 300

# time_bins = [1, 802.0, 4604.0, 81849.0, 1015617.0]
# 10 min, 1 hour, 12hours, 1 day, 1 week, max_time(11 days)
delta_t_bins = [1, 600.0, 3600.0, 43200.0, 86400, 604800, 1010249.0]
# 10 min, 1 hour,12 hours, 1 day, 1 week, 1 month,1 year, max_time(430 days)
delta_bins = [1, 600.0, 3600.0, 43200.0, 86400, 604800,  2592000, 31536000, 37246806.0]


n_time_bins = len(delta_bins) + 1  # 0 is used as padding
n_time_bins_t = len(delta_t_bins) + 1

def process_batch(batch, model):
    with torch.cuda.amp.autocast():

        skills = batch['skills'].to(device)
        correct = batch['correct'].to(device)
        attempts = batch['attempts'].to(device)
        format = batch['format'].to(device)
        wordsize = batch['wordsize'].to(device)
        # c_correct = batch['c_correct'].to(device)
        # c_attempts = batch['c_attempts'].to(device)

        delta = batch['delta'].to(device)
        delta_t = batch['delta_t'].to(device)
        delta_repeat = batch['delta_repeat'].to(device)
        time_bins = batch['time_bins'].to(device)
        time_bins_t = batch['time_bins_t'].to(device)
        targets = batch['targets'].to(device)
        mask = batch['mask'].to(device)
        if options.method == "dkt" :
            loss, outputs, true_count = model(skills, format, targets, mask)
        elif options.method == "dkt_plus" or options.method == "FIFAKT":
            loss, outputs, true_count,atttention = model(skills, format, wordsize, correct, attempts, delta, delta_t, delta_repeat, time_bins,time_bins_t, targets, mask)
        elif options.method == "dkt_wordsize" or options.method == "dkt_f":
            loss, outputs, true_count = model(skills, format, wordsize, correct, attempts, delta, delta_t, time_bins, time_bins_t,targets, mask)
        return targets, mask, loss, outputs, true_count, atttention



def evaluate(data_loader, model, return_outputs=False):
    # print(model.q_embed.weight.data, flush=True)
    # for n, p in model.named_parameters():
    #     print(n, p.data, flush=True)
    model.eval()
    with torch.no_grad():
        all_labels = None
        all_preds = None
        all_loss = None
        all_attention = None
        total_count = 0
        for batch in data_loader:
            targets, mask, loss, outputs, true_count, attention = process_batch(batch, model)
            # targets, mask, loss, outputs, true_count = process_batch_ini(batch, model)
            total_count += true_count

            mask = mask[:, 1:].contiguous().view(-1)
            masked_labels = targets[:, 1:].contiguous().view(-1)[mask]
            # mask = mask.view(-1)
            # masked_labels = targets.view(-1)[mask]
            masked_preds = outputs[mask]


            if all_labels is None:
                all_labels = masked_labels
                all_preds = masked_preds
                all_loss = loss.sum()

            else:
                all_labels = torch.cat((all_labels, masked_labels), dim=0)
                all_preds = torch.cat((all_preds, masked_preds), dim=0)
                all_loss += loss.sum()

            if all_attention is None:
                all_attention = attention

        all_labels = all_labels.cpu()
        all_preds = all_preds.cpu()

        out_loss = float(all_loss) / float(total_count)
        res = [elem for elem in np.abs(np.array(all_labels) - np.array(all_preds))]
        acc = 1 - sum(res) / len(res)

        # print(np.argwhere(np.isnan(all_preds.tolist())))

        auc = roc_auc_score(np.round(all_labels), all_preds)
        mae = mean_absolute_error(all_labels, all_preds)

        if return_outputs:
            return out_loss, acc, auc, mae, all_labels, all_preds, all_attention
        return out_loss, auc, mae


freeze_pretrained = options.freeze_embeddings
print("Should freeze embeddings?:", freeze_pretrained, flush=True)

# hyperparameters
real_batch_size = 256

if options.grid_search:
    hidden_dim_options = [64]
    final_fc_dim_options = [512]
    num_layer_options = [1]
    z_weight_options = [0.01]
    learning_rate_options = [5e-4]
    dropout_options = [0.2]
    batch_size_options = [real_batch_size]
    minority_weight_options = [1]
else:
    hidden_dim_options = [1024]
    final_fc_dim_options = [512]
    num_layer_options = [1]
    z_weight_options = [0.01]
    learning_rate_options = [5e-5]
    dropout_options = [0.2]
    batch_size_options = [real_batch_size]
    minority_weight_options = [5]

patience = 5

num_epochs = 200000

train_loader = DataLoader(train_dataset, batch_size=real_batch_size, shuffle=True, pin_memory=True, num_workers=0)
dev_loader = DataLoader(dev_dataset, batch_size=real_batch_size, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=real_batch_size, pin_memory=True)

train_batch_length = len(train_loader)
print("train batches:", train_batch_length, flush=True)
print("dev batches:", len(dev_loader), flush=True)
print("test batches:", len(test_loader), flush=True)

print("train masked size:", train_dataset.real_size(), flush=True)
print("dev masked size:", dev_dataset.real_size(), flush=True)
print("test masked size:", test_dataset.real_size(), flush=True)

best_model_auc = None
best_auc = None
best_parameters_auc = None
best_model_mae = None
best_mae = None
best_parameters_mae = None

for hidden_dim in hidden_dim_options:
    for final_fc_dim in final_fc_dim_options:
        for num_layers in num_layer_options:
            for learning_rate in learning_rate_options:
                for dropout in dropout_options:
                    for z_weight in z_weight_options:
                        for batch_size in batch_size_options:
                            for minority_weight in minority_weight_options:
                                current_parameters = (hidden_dim, final_fc_dim, num_layers, z_weight, learning_rate, dropout, batch_size, minority_weight)
                                print("-----new model parameters:", current_parameters, "-----", flush=True)

                                # class_weights = torch.from_numpy(np.load("./data/duolingo_hlr/class_distribution.npy"))
                                class_weights = torch.ones(10, dtype=torch.float32, device=device) * minority_weight
                                class_weights[-1] = 1
                                # print(class_weights, flush=True)

                                batches_needed_for_effective_size = int(batch_size / real_batch_size)
                                effective_num_batches = int(np.ceil(float(train_batch_length) / float(batches_needed_for_effective_size)))
                                print("effective number of training batches:", effective_num_batches, flush=True)

                                if options.method == "dkt":
                                    model = DKT(n_question, embed_dim, hidden_dim, layer_dim=num_layers, output_dim=num_layers, class_weights=class_weights, z_weight=z_weight)
                                elif options.method =="FIFAKT":
                                    model = FIFAKT(n_question, embed_dim, hidden_dim, layer_dim=num_layers, class_weights=class_weights, dropout=dropout, z_weight=z_weight, pretrained_embeddings=embeddings, freeze_pretrained=freeze_pretrained)
                                elif options.method == "dkt_f":
                                    model = DKT_f(n_question, embed_dim, hidden_dim, layer_dim=num_layers, output_dim=num_layers, class_weights=class_weights, z_weight=z_weight)
                                elif options.method == "dkt_plus":
                                    model = DKT_plus(n_question, embed_dim, n_time_bins, n_time_bins_t, hidden_dim,num_layers=num_layers, class_weights=class_weights,final_fc_dim=final_fc_dim, dropout=dropout, z_weight=z_weight, pretrained_embeddings=embeddings, freeze_pretrained=freeze_pretrained)
                                elif options.method == "dkt_wordsize":
                                    model = DKT_wordsize(n_question, embed_dim, n_time_bins, n_time_bins_t, hidden_dim,num_layers=num_layers, class_weights=class_weights,final_fc_dim=final_fc_dim, dropout=dropout, z_weight=z_weight,pretrained_embeddings=embeddings,freeze_pretrained=freeze_pretrained)
                                model.to(device)

                                print("model parameters:", count_parameters(model), flush=True)

                                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                                # start training
                                best_loss = None
                                patience_used = 0
                                start_time = time.time()

                                for epoch in range(num_epochs):
                                    batch_count = 0
                                    batch_size_counter = 0
                                    batch_total_count = 0
                                    model.train()
                                    for batch_i, batch in enumerate(train_loader):
                                        _, _, loss, outputs, true_count,_ = process_batch(batch, model)
                                        # _, _, loss, outputs, true_count = process_batch_ini(batch, model)
                                        loss = loss.mean()
                                        scaler.scale(loss).backward()
                                        batch_total_count += true_count

                                        batch_size_counter += 1
                                        if batch_size_counter == batches_needed_for_effective_size or batch_i == train_batch_length - 1:
                                            scaler.step(optimizer)
                                            scaler.update()
                                            optimizer.zero_grad()
                                            batch_size_counter = 0
                                            batch_total_count = 0

                                            batch_count += 1

                                            current_effective_batch = (batch_i + 1) / batches_needed_for_effective_size

                                            # if current_effective_batch % int(float(effective_num_batches) / 4.0) == 0:
                                            #     print("batch", current_effective_batch, "/", effective_num_batches, ", loss:", float(loss), ", average time per batch:", (time.time() - start_time) / batch_count, flush=True)
                                            #     start_time = time.time()
                                            #     batch_count = 0

                                    dev_loss, dev_auc, dev_mae = evaluate(dev_loader, model)

                                    dev_loss = float(dev_loss)



                                    elapsed = (time.time() - start_time)
                                    print("epoch", epoch, "dev loss:", dev_loss, "patience used:", patience_used, "/", patience, "dev auc:", dev_auc, "dev mae:", dev_mae, ",", elapsed, "s elapsed per epoch",  flush=True)
                                    start_time = time.time()

                                    if best_loss is None or dev_loss < best_loss:
                                        patience_used = 0
                                        best_loss = dev_loss
                                    else:
                                        patience_used += 1
                                        if patience_used >= patience:
                                            break

                                    if best_auc is None or dev_auc > best_auc:
                                        best_auc = dev_auc
                                        best_model_auc = copy.deepcopy(model)
                                        best_parameters_auc = current_parameters

                                    if best_mae is None or dev_mae < best_mae:
                                        best_mae = dev_mae
                                        best_model_mae = copy.deepcopy(model)
                                        best_parameters_mae = current_parameters

                            test_loss, test_acc, test_auc, test_mae, all_labels, all_preds, attention = evaluate(test_loader, model, return_outputs=True)
                            print("test loss:", test_loss, "test acc:", test_acc, "test auc:", test_auc, "test mae:", test_mae, flush=True)
                            # print("attention:", atttention)
                            # np.savetxt('atttention_tagetomo.csv', attention.cpu().numpy().reshape((-1,199)))


def save_model_and_predictions(model, all_labels, all_preds, out_modifier=""):
    # output predictions
    predictions_filename = options.out_filename
    if out_modifier != "":
        predictions_filename = predictions_filename.replace(".pt", ".predictions_" + out_modifier + ".csv")
    else:
        predictions_filename = predictions_filename.replace(".pt", ".predictions.csv")

    with open(predictions_filename, "w") as predictions_file:
        predictions_file.write("y,y_hat\n")
        for y, y_hat in zip(all_labels.tolist(), all_preds.tolist()):
            predictions_file.write(str(y) + "," + str(float(y_hat)) + "\n")

    model = model.cpu()

    model_filename = options.out_filename
    if out_modifier != "":
        model_filename = model_filename.replace(".pt", "." + out_modifier + ".pt")

    torch.save(model, model_filename)


if options.grid_search:
    print("---best model AUC:", best_auc, "parameters:", best_parameters_auc, "---", flush=True)
    model = best_model_auc
    model = model.to(device)

    test_loss,test_aac, test_auc, test_mae, all_labels, all_preds,_ = evaluate(test_loader, model, return_outputs=True)
    print("test loss:", test_loss,"test acc:", test_acc, "test auc:", test_auc, "test mae:", test_mae, flush=True)

    save_model_and_predictions(model, all_labels, all_preds, out_modifier="best_auc")

    print("---best model MAE:", best_mae, "parameters:", best_parameters_mae, "---", flush=True)
    model = best_model_mae
    model = model.to(device)

    test_loss,test_aac, test_auc, test_mae, all_labels, all_preds,_ = evaluate(test_loader, model, return_outputs=True)
    print("test loss:", test_loss,"test acc:", test_acc, "test auc:", test_auc, "test mae:", test_mae, flush=True)

    save_model_and_predictions(model, all_labels, all_preds, out_modifier="best_mae")

else:
    save_model_and_predictions(model, all_labels, all_preds)
