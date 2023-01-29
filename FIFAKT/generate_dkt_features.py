import numpy as np
import argparse
import pandas as pd

parser.add_argument('--data', type=str, nargs='?', default="../duolingo.csv")
parser.add_argument('--emb_file', type=str, nargs='?', default='all_word_embeddings_fastword.csv')
# parser.add_argument('--kc_emb_file', type=str, nargs='?', default='en_kc_embeddings.csv')
parser.add_argument('--max_seq_len', type=int, nargs='?', default=200)
options = parser.parse_args()

########################################################
########################################################

print("reading embeddings...", flush=True)

emb_key = {}
with open(options.emb_file, "r", encoding="utf-8") as emb_file:
    for line in emb_file:
        emb_word, emb_lang, embedding = line.strip().split("\t")
        emb_key[emb_word + "\t" + emb_lang] = len(emb_key) + 1  # add one because index 0 is reserved for padding


# kc_emb_key = {}
# with open(options.kc_emb_file, "r", encoding="utf-8") as kc_emb_file:
#     for line in kc_emb_file:
#         emb_lemma, emb_lang, embedding = line.strip().split("\t")
#         kc_emb_key[emb_lemma + "\t" + emb_lang] = len(kc_emb_key) + 1  # add one because index 0 is reserved for padding


print("mapped word/lang to embeddings", flush=True)

init_time = None

# max delta_t = 1010249(11 days), max delta= 37246806(430 days)
# bins = [1, 802.0, 4604.0, 81849.0, 1015617.0]
# 10 min, 1 hour, 12hours, 1 day, 1 week, max_time(11 days)
delta_bins = [0, 600.0, 3600.0, 43200.0, 86400, 604800, 1010249.0, 38776657.0]
# 10 min, 1 hour,12 hours, 1 day, 1 week, 1 month,1 year, max_time(430 days)
delta_t_bins = [0, 600.0, 3600.0, 43200.0, 86400, 604800,  2592000, 31536000, 48776657.0]
total_num_lines = 5014790

dev_cutoff = int(total_num_lines * 0.8)
test_cutoff = int(total_num_lines * 0.9)

print("dev cutoff:", dev_cutoff, flush=True)
print("test cutoff:", test_cutoff, flush=True)

users = {}
items = {}
langs = {}
lans = {}

items_by_user = {}
langs_by_user = {}
lans_by_user = {}
user_by_user = {}

# kcs_by_user = {}
skills_by_user = {}
p_by_user = {}
timestamp_by_user = {}
delta_by_user = {}
delta_t_by_user = {}
delta_s_by_user = {}
time_bin_by_user = {}
time_bin_t_by_user = {}
seen_by_user = {}
c_seen_by_user = {}
correct_by_user = {}
c_correct_by_user = {}

eval_start_by_user = {}
wordsize_by_user = {}

last_time = {}
word_last_time = {}

print("beginning to process data...", flush=True)

with open(options.data, encoding="utf-8") as data_file:
    data_file.readline()
    for i, line in enumerate(data_file):
        p_recall, timestamp, delta, user_id, learning_language, ui_language, lexeme_id, lexeme_string, history_seen, history_correct, session_seen, session_correct = line.strip().split(",")

        delta = int(delta)


        timestamp = float(timestamp)
        if init_time is None:
            init_time = timestamp
        timestamp -= init_time
        assert timestamp >= 0

        ## sequence time gap
        if user_id in last_time:
            delta_t = timestamp - last_time[user_id]
        else:
            delta_t = 0

        if delta_t < 0:
            print("invalid delta_t:", delta_t, "line", i, "time:", timestamp, "last_time:", last_time[user_id], flush=True)
            if delta_t > -5:
                delta_t = 0
                timestamp = last_time[user_id]
        assert delta_t >= 0

        last_time[user_id] = timestamp

        time_bin = None
        for k, b in enumerate(delta_bins):
            if delta_t <= b:
                time_bin = k + 1
                break

        time_bin_t = None
        for k, b in enumerate(delta_t_bins):
            if delta <= b:
                time_bin_t = k + 1
                break

        if time_bin is None or time_bin_t is None:
            print("invalid delta_t for timebins:", delta, time_bin, delta_t,time_bin_t, flush=True)
            exit()

        p_recall = float(p_recall)

        # history_seen = int(history_seen)
        # history_correct = int(history_correct)

        history_seen = np.log(int(history_seen))
        history_correct = np.log(int(history_correct))

        session_seen = int(session_seen)
        session_correct = int(session_correct)

        word, remainder = lexeme_string.split("/", 1)
        if word == "<*sf>":
            word = remainder.split("<", 1)[0]

        lemma = lexeme_string.split("/")[1]

        ## sameword time gap
        if str(user_id)+word in word_last_time:
            delta_s = timestamp - word_last_time[str(user_id)+word]
        else:
            delta_s = 0

        assert delta_s >= 0
        word_last_time[str(user_id)+word] = timestamp


        wordsize = len(word)
        # kc_id = kc_emb_key[lemma + "\t" + learning_language]
        skill_id = emb_key[word + "\t" + learning_language]
        ## sameword time gap
        delta_s = round(delta_s / 86400, 3)
        ## sequence time gap
        delta_t = round(delta_t / 86400, 3)
        ## repeat time gap
        delta = round(delta / 86400, 3)

        user_key = user_id
        if i >= test_cutoff:
            user_key += "_test"
        elif i >= dev_cutoff:
            user_key += "_dev"

        if user_key not in skills_by_user:
            if i >= test_cutoff and (user_id + "_dev") in skills_by_user:
                skills_by_user[user_key] = skills_by_user[user_id + "_dev"][:]
                # kcs_by_user[user_key] = kcs_by_user[user_id + "_dev"][:]
                p_by_user[user_key] = p_by_user[user_id + "_dev"][:]
                timestamp_by_user[user_key] = timestamp_by_user[user_id + "_dev"][:]
                time_bin_t_by_user[user_key] = time_bin_t_by_user[user_id + "_dev"][:]
                time_bin_by_user[user_key] = time_bin_by_user[user_id + "_dev"][:]
                seen_by_user[user_key] = seen_by_user[user_id + "_dev"][:]
                correct_by_user[user_key] = correct_by_user[user_id + "_dev"][:]
                c_seen_by_user[user_key] = c_seen_by_user[user_id + "_dev"][:]
                c_correct_by_user[user_key] = c_correct_by_user[user_id + "_dev"][:]
                user_by_user[user_key] = user_by_user[user_id + "_dev"]
                items_by_user[user_key] = items_by_user[user_id + "_dev"][:]
                langs_by_user[user_key] = langs_by_user[user_id + "_dev"][:]
                lans_by_user[user_key] = lans_by_user[user_id + "_dev"][:]
                delta_by_user[user_key] = delta_by_user[user_id + "_dev"][:]
                delta_t_by_user[user_key] = delta_t_by_user[user_id + "_dev"][:]
                delta_s_by_user[user_key] = delta_s_by_user[user_id + "_dev"][:]
                wordsize_by_user[user_key] = wordsize_by_user[user_id + "_dev"][:]

            elif i >= dev_cutoff and user_id in skills_by_user:
                skills_by_user[user_key] = skills_by_user[user_id][:]
                # kcs_by_user[user_key] = kcs_by_user[user_id][:]
                p_by_user[user_key] = p_by_user[user_id][:]
                timestamp_by_user[user_key] = timestamp_by_user[user_id][:]
                time_bin_by_user[user_key] = time_bin_by_user[user_id][:]
                time_bin_t_by_user[user_key] = time_bin_t_by_user[user_id][:]
                seen_by_user[user_key] = seen_by_user[user_id][:]
                correct_by_user[user_key] = correct_by_user[user_id][:]
                c_seen_by_user[user_key] = c_seen_by_user[user_id][:]
                c_correct_by_user[user_key] = c_correct_by_user[user_id][:]
                user_by_user[user_key] = user_by_user[user_id]
                items_by_user[user_key] = items_by_user[user_id][:]
                langs_by_user[user_key] = langs_by_user[user_id][:]
                lans_by_user[user_key] = lans_by_user[user_id][:]
                delta_by_user[user_key] = delta_by_user[user_id][:]
                delta_t_by_user[user_key] = delta_t_by_user[user_id][:]
                delta_s_by_user[user_key] = delta_s_by_user[user_id][:]
                wordsize_by_user[user_key] = wordsize_by_user[user_id][:]

            else:
                skills_by_user[user_key] = []
                # kcs_by_user[user_key] = []
                p_by_user[user_key] = []
                timestamp_by_user[user_key] = []
                time_bin_by_user[user_key] = []
                time_bin_t_by_user[user_key] = []
                seen_by_user[user_key] = []
                c_seen_by_user[user_key] = []
                correct_by_user[user_key] = []
                c_correct_by_user[user_key] = []
                delta_by_user[user_key] = []
                delta_t_by_user[user_key] = []
                delta_s_by_user[user_key] = []

                if user_id not in users:
                    users[user_id] = len(users)
                user_by_user[user_key] = []


                items_by_user[user_key] = []
                langs_by_user[user_key] = []
                lans_by_user[user_key] = []
                wordsize_by_user[user_key] = []

            eval_start_by_user[user_key] = len(skills_by_user[user_key])  # start postion for dev/test

        skills_by_user[user_key].append(skill_id)
        # kcs_by_user[user_key].append(kc_id)
        p_by_user[user_key].append(p_recall)
        timestamp_by_user[user_key].append(timestamp)
        time_bin_by_user[user_key].append(time_bin)
        time_bin_t_by_user[user_key].append(time_bin_t)
        seen_by_user[user_key].append(history_seen)
        c_seen_by_user[user_key].append(session_seen)
        correct_by_user[user_key].append(history_correct)
        c_correct_by_user[user_key].append(session_seen)
        user_by_user[user_key].append(users[user_id])
        delta_by_user[user_key].append(delta)
        delta_t_by_user[user_key].append(delta_t)
        delta_s_by_user[user_key].append(delta_s)
        wordsize_by_user[user_key].append(wordsize)

        if lexeme_id not in items:
            items[lexeme_id] = len(items)
        items_by_user[user_key].append(items[lexeme_id])
        combined_lang = ui_language + "->" + learning_language
        if combined_lang not in langs:
            langs[combined_lang] = len(langs)
        langs_by_user[user_key].append(langs[combined_lang])

        lan = learning_language
        if lan not in lans:
            lans[lan] = len(lans)
        lans_by_user[user_key].append(lans[lan])

        if i % 100000 == 0:
            print(i, "processed", flush=True)

all_user_keys = list(skills_by_user.keys())
print("user number",len(users))
######------------------train set-----------------############

print("building train set...", flush=True)

out_file_modifier = "en"

# train set
all_users = []
all_items = []
all_langs = []
all_lans = []
all_skills = []
# all_kcs = []
all_p = []
all_timestamp = []
all_delta = []
all_delta_t = []
all_delta_s = []
all_time_bin = []
all_time_bin_t = []
all_seen = []
all_correct = []
all_c_seen = []
all_c_correct = []
eval_mask = []
all_wordsize = []

seq_lens = []

split_counter = 0

for u in all_user_keys:
    if "_dev" not in u and "_test" not in u:
        seq_len = len(skills_by_user[u])
        seq_lens.append(seq_len)
        cursor = options.max_seq_len
        eval_start = eval_start_by_user[u]
        while cursor < seq_len:
            all_users.append(user_by_user[u][cursor - options.max_seq_len: cursor])
            all_items.append(items_by_user[u][cursor - options.max_seq_len: cursor])
            all_langs.append(langs_by_user[u][cursor - options.max_seq_len: cursor])
            all_lans.append(lans_by_user[u][cursor - options.max_seq_len: cursor])
            all_delta.append(delta_by_user[u][cursor - options.max_seq_len: cursor])
            all_delta_t.append(delta_t_by_user[u][cursor - options.max_seq_len: cursor])
            all_delta_s.append(delta_s_by_user[u][cursor - options.max_seq_len: cursor])

            all_skills.append(skills_by_user[u][cursor - options.max_seq_len: cursor])
            # all_kcs.append(kcs_by_user[u][cursor - options.max_seq_len: cursor])
            all_p.append(p_by_user[u][cursor - options.max_seq_len: cursor])
            all_timestamp.append(timestamp_by_user[u][cursor - options.max_seq_len: cursor])
            all_time_bin.append(time_bin_by_user[u][cursor - options.max_seq_len: cursor])
            all_time_bin_t.append(time_bin_by_user[u][cursor - options.max_seq_len: cursor])
            all_seen.append(seen_by_user[u][cursor - options.max_seq_len: cursor])
            all_correct.append(correct_by_user[u][cursor - options.max_seq_len: cursor])
            all_c_seen.append(c_seen_by_user[u][cursor - options.max_seq_len: cursor])
            all_c_correct.append(c_correct_by_user[u][cursor - options.max_seq_len: cursor])
            all_wordsize.append(wordsize_by_user[u][cursor - options.max_seq_len: cursor])

            mask = np.zeros(options.max_seq_len)
            adjusted_mask_start = np.maximum(0, eval_start - (cursor - options.max_seq_len))
            if adjusted_mask_start < len(mask):
                mask[adjusted_mask_start:] = 1
            eval_mask.append(list(mask))

            cursor += options.max_seq_len
            split_counter += 1

        # lower_bound = np.maximum(0, seq_len - options.max_seq_len)
        # padding_needed = options.max_seq_len - (seq_len - lower_bound)
        lower_bound = np.maximum(0, cursor - options.max_seq_len)
        padding_needed = options.max_seq_len - (seq_len - lower_bound)
        all_users.append(user_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_items.append(items_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_langs.append(langs_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_lans.append(lans_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_delta.append(delta_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_delta_t.append(delta_t_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_delta_s.append(delta_s_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))

        all_skills.append(skills_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        # all_kcs.append(kcs_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_p.append(p_by_user[u][lower_bound: seq_len] + ([-1] * padding_needed))
        all_timestamp.append(timestamp_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_time_bin.append(time_bin_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_time_bin_t.append(time_bin_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_seen.append(seen_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_correct.append(correct_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_c_seen.append(c_seen_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_c_correct.append(c_correct_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_wordsize.append(wordsize_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))

        mask = np.zeros(seq_len - lower_bound)
        adjusted_mask_start = np.maximum(0, eval_start - lower_bound)
        if adjusted_mask_start < len(mask):
            mask[adjusted_mask_start:] = 1
        # if cursor > options.max_seq_len:
        #     already_considered_mask = np.zeros(seq_len - lower_bound)
        #     end_considered = (cursor - options.max_seq_len) - lower_bound
        #     already_considered_mask[end_considered:] = 1
        #     mask *= already_considered_mask
        eval_mask.append(list(mask) + ([0] * padding_needed))

all_skills = np.array(all_skills)
np.save("train_skills" + out_file_modifier, all_skills)
# np.save("train_kcs" + out_file_modifier, all_kcs)
np.save("train_users" + out_file_modifier, np.array(all_users))
np.save("train_items" + out_file_modifier, np.array(all_items))
np.save("train_langs" + out_file_modifier, np.array(all_langs))
np.save("train_lans" + out_file_modifier, np.array(all_lans))
all_p = np.array(all_p)
np.save("train_p" + out_file_modifier, all_p)
np.save("train_delta" + out_file_modifier, np.array(all_delta))
np.save("train_delta_t" + out_file_modifier, np.array(all_delta_t))
np.save("train_delta_s" + out_file_modifier, np.array(all_delta_s))
np.save("train_timestamp" + out_file_modifier, np.array(all_timestamp))
np.save("train_time_bin" + out_file_modifier, np.array(all_time_bin))
np.save("train_time_bin_t" + out_file_modifier, np.array(all_time_bin_t))
np.save("train_seen" + out_file_modifier, np.array(all_seen))
np.save("train_correct" + out_file_modifier, np.array(all_correct))
np.save("train_c_seen" + out_file_modifier, np.array(all_c_seen))
np.save("train_c_correct" + out_file_modifier, np.array(all_c_correct))
eval_mask = np.array(eval_mask, dtype=np.int32)
np.save("train_eval_mask" + out_file_modifier, eval_mask)
np.save("train_wordsize" + out_file_modifier, np.array(all_wordsize))

print("train shape:", np.shape(all_skills), flush=True)
print("train size:", np.sum(eval_mask), flush=True)
print("times a user was split:", split_counter, flush=True)

print(all_skills, flush=True)
# print(all_p, flush=True)
# print(all_p != -1, flush=True)
# print(eval_mask, flush=True)
padding_needed = np.sum(all_skills == 0, axis=1)
print("avg padding:", np.mean(padding_needed), "median padding:", np.median(padding_needed), "min padding:", np.min(padding_needed), "max padding:", np.max(padding_needed), flush=True)
print("avg seq_len:", np.mean(seq_lens), "median seq_len:", np.median(seq_lens), "min seq_len:", np.min(seq_lens), "max seq_len:", np.max(seq_lens), flush=True)

######------------------dev set-----------------############

print("building dev set...", flush=True)

# dev set
all_users = []
all_items = []
all_langs = []
all_lans = []
all_skills = []
# all_kcs = []
all_p = []
all_timestamp = []
all_delta = []
all_delta_t = []
all_delta_s = []
all_time_bin = []
all_time_bin_t = []
all_seen = []
all_correct = []
all_c_seen = []
all_c_correct = []
eval_mask = []
all_wordsize = []

split_counter = 0

for u in all_user_keys:
    if "_dev" in u:
        seq_len = len(skills_by_user[u])
        cursor = options.max_seq_len
        eval_start = eval_start_by_user[u]
        while cursor < seq_len:
            all_users.append(user_by_user[u][cursor - options.max_seq_len: cursor])
            all_items.append(items_by_user[u][cursor - options.max_seq_len: cursor])
            all_langs.append(langs_by_user[u][cursor - options.max_seq_len: cursor])
            all_lans.append(lans_by_user[u][cursor - options.max_seq_len: cursor])
            all_delta.append(delta_by_user[u][cursor - options.max_seq_len: cursor])
            all_delta_t.append(delta_t_by_user[u][cursor - options.max_seq_len: cursor])
            all_delta_s.append(delta_s_by_user[u][cursor - options.max_seq_len: cursor])


            all_skills.append(skills_by_user[u][cursor - options.max_seq_len: cursor])
            # all_kcs.append(kcs_by_user[u][cursor - options.max_seq_len: cursor])
            all_p.append(p_by_user[u][cursor - options.max_seq_len: cursor])
            all_timestamp.append(timestamp_by_user[u][cursor - options.max_seq_len: cursor])
            all_time_bin.append(time_bin_by_user[u][cursor - options.max_seq_len: cursor])
            all_time_bin_t.append(time_bin_by_user[u][cursor - options.max_seq_len: cursor])
            all_seen.append(seen_by_user[u][cursor - options.max_seq_len: cursor])
            all_correct.append(correct_by_user[u][cursor - options.max_seq_len: cursor])
            all_c_seen.append(c_seen_by_user[u][cursor - options.max_seq_len: cursor])
            all_c_correct.append(c_correct_by_user[u][cursor - options.max_seq_len: cursor])
            all_wordsize.append(wordsize_by_user[u][cursor - options.max_seq_len: cursor])

            mask = np.zeros(options.max_seq_len)
            adjusted_mask_start = np.maximum(0, eval_start - (cursor - options.max_seq_len))
            if adjusted_mask_start < len(mask):
                mask[adjusted_mask_start:] = 1
            eval_mask.append(list(mask))

            cursor += options.max_seq_len
            split_counter += 1

        lower_bound = np.maximum(0, cursor - options.max_seq_len)
        padding_needed = options.max_seq_len - (seq_len - lower_bound)
        all_users.append(user_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_items.append(items_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_langs.append(langs_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_lans.append(lans_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_delta.append(delta_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_delta_t.append(delta_t_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_delta_s.append(delta_s_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))

        all_skills.append(skills_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        # all_kcs.append(kcs_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_p.append(p_by_user[u][lower_bound: seq_len] + ([-1] * padding_needed))
        all_timestamp.append(timestamp_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_time_bin.append(time_bin_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_time_bin_t.append(time_bin_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_seen.append(seen_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_correct.append(correct_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_c_seen.append(c_seen_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_c_correct.append(c_correct_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_wordsize.append(wordsize_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))

        mask = np.zeros(seq_len - lower_bound)
        adjusted_mask_start = np.maximum(0, eval_start - lower_bound)   #train data mask
        if adjusted_mask_start < len(mask):
            mask[adjusted_mask_start:] = 1
        # if cursor > options.max_seq_len:
        #     already_considered_mask = np.zeros(seq_len - lower_bound)
        #     end_considered = (cursor - options.max_seq_len) - lower_bound
        #     already_considered_mask[end_considered:] = 1
        #     mask *= already_considered_mask
        eval_mask.append(list(mask) + ([0] * padding_needed))

all_skills = np.array(all_skills)
np.save("dev_skills" + out_file_modifier, all_skills)
# np.save("dev_kcs" + out_file_modifier, all_kcs)

np.save("dev_users" + out_file_modifier, np.array(all_users))
np.save("dev_items" + out_file_modifier, np.array(all_items))
np.save("dev_langs" + out_file_modifier, np.array(all_langs))
np.save("dev_lans" + out_file_modifier, np.array(all_lans))
np.save("dev_delta" + out_file_modifier, np.array(all_delta))
np.save("dev_delta_t" + out_file_modifier, np.array(all_delta_t))
np.save("dev_delta_s" + out_file_modifier, np.array(all_delta_s))
all_p = np.array(all_p)
np.save("dev_p" + out_file_modifier, all_p)
np.save("dev_timestamp" + out_file_modifier, np.array(all_timestamp))
np.save("dev_time_bin" + out_file_modifier, np.array(all_time_bin))
np.save("dev_time_bin_t" + out_file_modifier, np.array(all_time_bin_t))
np.save("dev_seen" + out_file_modifier, np.array(all_seen))
np.save("dev_correct" + out_file_modifier, np.array(all_correct))
np.save("dev_c_seen" + out_file_modifier, np.array(all_c_seen))
np.save("dev_c_correct" + out_file_modifier, np.array(all_c_correct))
eval_mask = np.array(eval_mask, dtype=np.int32)
np.save("dev_eval_mask" + out_file_modifier, eval_mask)
np.save("dev_wordsize" + out_file_modifier, np.array(all_wordsize))

print("dev shape:", np.shape(all_skills), flush=True)
# print("p size:", np.shape(all_p), "eval_mask size:", np.shape(eval_mask), flush=True)
print("dev size:", np.sum(eval_mask), flush=True)
print("times a user was split:", split_counter, flush=True)

######------------------test set-----------------############
print("building test set...", flush=True)

# test set
all_users = []
all_items = []
all_langs = []
all_lans = []
all_skills = []
# all_kcs = []
all_p = []
all_timestamp = []
all_time_bin = []
all_time_bin_t = []
all_delta = []
all_delta_t = []
all_delta_s = []
all_seen = []
all_correct = []
all_c_seen = []
all_c_correct = []
eval_mask = []
all_wordsize = []
split_counter = 0

for u in all_user_keys:
    if "_test" in u:
        seq_len = len(skills_by_user[u])
        cursor = options.max_seq_len
        eval_start = eval_start_by_user[u]
        while cursor < seq_len:
            all_users.append(user_by_user[u][cursor - options.max_seq_len: cursor])
            all_items.append(items_by_user[u][cursor - options.max_seq_len: cursor])
            all_langs.append(langs_by_user[u][cursor - options.max_seq_len: cursor])
            all_lans.append(lans_by_user[u][cursor - options.max_seq_len: cursor])
            all_delta.append(delta_by_user[u][cursor - options.max_seq_len: cursor])
            all_delta_t.append(delta_t_by_user[u][cursor - options.max_seq_len: cursor])
            all_delta_s.append(delta_s_by_user[u][cursor - options.max_seq_len: cursor])

            all_skills.append(skills_by_user[u][cursor - options.max_seq_len: cursor])
            # all_kcs.append(kcs_by_user[u][cursor - options.max_seq_len: cursor])
            all_p.append(p_by_user[u][cursor - options.max_seq_len: cursor])
            all_timestamp.append(timestamp_by_user[u][cursor - options.max_seq_len: cursor])
            all_time_bin.append(time_bin_by_user[u][cursor - options.max_seq_len: cursor])
            all_time_bin_t.append(time_bin_by_user[u][cursor - options.max_seq_len: cursor])
            all_seen.append(seen_by_user[u][cursor - options.max_seq_len: cursor])
            all_correct.append(correct_by_user[u][cursor - options.max_seq_len: cursor])
            all_c_seen.append(c_seen_by_user[u][cursor - options.max_seq_len: cursor])
            all_c_correct.append(c_correct_by_user[u][cursor - options.max_seq_len: cursor])
            all_wordsize.append(wordsize_by_user[u][cursor - options.max_seq_len: cursor])

            mask = np.zeros(options.max_seq_len)
            adjusted_mask_start = np.maximum(0, eval_start - (cursor - options.max_seq_len))
            if adjusted_mask_start < len(mask):
                mask[adjusted_mask_start:] = 1
            eval_mask.append(list(mask))

            cursor += options.max_seq_len
            split_counter += 1

        lower_bound = np.maximum(0, cursor - options.max_seq_len)
        padding_needed = options.max_seq_len - (seq_len - lower_bound)
        all_users.append(user_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_items.append(items_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_langs.append(langs_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_lans.append(lans_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_delta.append(delta_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_delta_t.append(delta_t_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_delta_s.append(delta_s_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))

        all_skills.append(skills_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        # all_kcs.append(kcs_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_p.append(p_by_user[u][lower_bound: seq_len] + ([-1] * padding_needed))
        all_timestamp.append(timestamp_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_time_bin.append(time_bin_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_time_bin_t.append(time_bin_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_seen.append(seen_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_correct.append(correct_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_c_seen.append(c_seen_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_c_correct.append(c_correct_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))
        all_wordsize.append(wordsize_by_user[u][lower_bound: seq_len] + ([0] * padding_needed))

        mask = np.zeros(seq_len - lower_bound)
        adjusted_mask_start = np.maximum(0, eval_start - lower_bound)
        if adjusted_mask_start < len(mask):
            mask[adjusted_mask_start:] = 1
        # if cursor > options.max_seq_len:
        #     already_considered_mask = np.zeros(seq_len - lower_bound)
        #     end_considered = (cursor - options.max_seq_len) - lower_bound
        #     already_considered_mask[end_considered:] = 1
        #     mask *= already_considered_mask
        eval_mask.append(list(mask) + ([0] * padding_needed))

all_skills = np.array(all_skills)
np.save("test_skills" + out_file_modifier, all_skills)
# np.save("test_kcs" + out_file_modifier, all_kcs)
np.save("test_users" + out_file_modifier, np.array(all_users))
np.save("test_items" + out_file_modifier, np.array(all_items))
np.save("test_langs" + out_file_modifier, np.array(all_langs))
np.save("test_lans" + out_file_modifier, np.array(all_langs))
np.save("test_delta" + out_file_modifier, np.array(all_delta))
np.save("test_delta_t" + out_file_modifier, np.array(all_delta_t))
np.save("test_delta_s" + out_file_modifier, np.array(all_delta_s))
all_p = np.array(all_p)
np.save("test_p" + out_file_modifier, all_p)
np.save("test_timestamp" + out_file_modifier, np.array(all_timestamp))
np.save("test_time_bin" + out_file_modifier, np.array(all_time_bin))
np.save("test_time_bin_t" + out_file_modifier, np.array(all_time_bin_t))
np.save("test_seen" + out_file_modifier, np.array(all_seen))
np.save("test_correct" + out_file_modifier, np.array(all_correct))
np.save("test_c_seen" + out_file_modifier, np.array(all_c_seen))
np.save("test_c_correct" + out_file_modifier, np.array(all_c_correct))
eval_mask = np.array(eval_mask, dtype=np.int32)
np.save("test_eval_mask" + out_file_modifier, eval_mask)
np.save("test_wordsize" + out_file_modifier, np.array(all_wordsize))

print("test shape:", np.shape(all_skills), flush=True)
# print("p size:", np.shape(all_p), "eval_mask size:", np.shape(eval_mask), flush=True)
print("test size:", np.sum(eval_mask), flush=True)
print("times a user was split:", split_counter, flush=True)
