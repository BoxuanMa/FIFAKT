import numpy as np
import pandas as pd
from scipy import sparse
import argparse
import os
import sentencepiece as spm
import glob
# import json

parser = argparse.ArgumentParser(description='Prepare datasets.')
parser.add_argument('--dataset', type=str, nargs='?', default='tagetomo') #tagetomo duolingo_hlr
parser.add_argument('--min_interactions', type=int, nargs='?', default=0)
parser.add_argument('--remove_nan_skills', type=bool, nargs='?', const=True, default=False)

parser.add_argument('--continuous_correct', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--tags', type=bool, nargs='?', const=True, default=True)
parser.add_argument('--lemma', type=bool, nargs='?', const=True, default=True)
parser.add_argument('--subword_skills', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--tokenizer_dir', type=str, nargs='?', default='./vocab/')
parser.add_argument('--vocab_size', type=int, nargs='?', default=5000)
parser.add_argument('--nbest', type=int, nargs='?', default=2)
options = parser.parse_args()



lang_tokenizers = {}
os.chdir('E:/project/src/das3h/vocab')

if options.subword_skills:
    print("Loading sentence piece tokenizers...", flush=True)
    for path in glob.glob("**" + "_vocab=" + str(options.vocab_size) + "**.model"):
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(path)
        if options.dataset == 'duolingo_hlr':
            lang = path.split("/")[-1].split("_")[0]
        else:
            lang = 'en1900'
            print('lang = en1900')
        lang_tokenizers[lang] = sp_model
        print(lang)

def extract_subword_skills(word_df, language_df, nbest=1, tags=False, include_lemma=False):
    skill_df = []
    for i in range(len(word_df)):
        word = word_df[i].split("/")[0]
        if word == "<*sf>":
            word = word_df[i].split("/")[1].split("<")[0]
        lang = language_df[i]
        nbest_segmentations = lang_tokenizers[lang].NBestEncodeAsPieces(word, nbest)

        piece_set = set()
        for segmentation in nbest_segmentations:
            if word not in segmentation:
                piece_set.update(segmentation)

        if tags:
            all_tags = word_df[i].split("/")[1].split("<")[1:]
            for tag in all_tags:
                piece_set.add("<"+tag)
        if include_lemma:
            piece_set.add("▁" + word_df[i].split("/")[1].split("<")[0])

        skill_set = "~~".join(map(lambda x: lang + ":" + x, piece_set))

        skill_df.append(skill_set)

    return skill_df


def extract_subword_skills_tage(word_df, nbest=1, tags=False, include_lemma=False):
    skill_df = []
    for i in range(len(word_df)):
        word = word_df[i]
        lang = 'en1900'
        nbest_segmentations = lang_tokenizers[lang].NBestEncodeAsPieces(word, nbest)

        piece_set = set()
        for segmentation in nbest_segmentations:
            if word not in segmentation:
                piece_set.update(segmentation)

        if tags:
            all_tags = word_df[i].split("/")[1].split("<")[1:]
            for tag in all_tags:
                piece_set.add("<"+tag)
        if include_lemma:
            piece_set.add("▁" + word_df[i].split("/")[1].split("<")[0])

        skill_set = "~~".join(map(lambda x: lang + ":" + x, piece_set))

        skill_df.append(skill_set)

    return skill_df


def extract_tag_skills(word_df, language_df, lemma=False):
    skill_df = []

    for i in range(len(word_df)):
        word = word_df[i].split("/")[0]
        if word == "<*sf>":
            word = word_df[i].split("/")[1].split("<")[0]
        lang = language_df[i]

        piece_set = set()

        if not lemma:
            piece_set.add(word)

        all_tags = word_df[i].split("/")[1].split("<")
        for t in range(len(all_tags)):
            new_skill = all_tags[t]
            if t == 0:
                if lemma:
                    piece_set.add(new_skill)
            else:
                piece_set.add("<" + new_skill)

        skill_set = "~~".join(map(lambda x: lang + ":" + x, piece_set))

        skill_df.append(skill_set)

    return skill_df

def extract_tag_skills_tage(word_df,  lemma=False):
    skill_df = []

    for i in range(len(word_df)):
        word = word_df[i].split("/")[0]

        lang = 'en1900'

        piece_set = set()

        if not lemma:
            piece_set.add(word)

        all_tags = word_df[i].split("/")[1].split("<")
        for t in range(len(all_tags)):
            new_skill = all_tags[t]
            if t == 0:
                if lemma:
                    piece_set.add(new_skill)
            else:
                piece_set.add("<" + new_skill)

        skill_set = "~~".join(map(lambda x: lang + ":" + x, piece_set))

        skill_df.append(skill_set)


    return skill_df

def prepare_duolingo_hlr(min_interactions_per_user, remove_nan_skills, subword_skills=False, tags=False, lemma=False, drop_duplicates=False):
    """Preprocess dataset released with their Half-Life Regression model.

    Arguments:
    min_interactions_per_user -- minimum number of interactions per student
    remove_nan_skills -- if True, remove interactions with no skill tag

    Outputs:
    df -- preprocessed Duolingo dataset (pandas DataFrame)
    Q_mat -- corresponding q-matrix (item-skill relationships sparse array)
    """
    df = pd.read_csv("E:/project/src/duolingo_en.csv", encoding="utf-8")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()

    df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)


    if subword_skills:
        df["skill_id"] = extract_subword_skills(df["lexeme_string"], df["learning_language"], nbest=options.nbest, include_lemma=lemma, tags=tags)
    elif tags:
        df["skill_id"] = extract_tag_skills(df["lexeme_string"], df["learning_language"], lemma=lemma)

    elif lemma:
        df["skill_id"] = df["learning_language"] + ":" + df["lexeme_string"].apply(lambda x: x.split("/")[1].split("<")[0])
    else:
        df["skill_id"] = df["learning_language"] + ":" + df["lexeme_string"]  # .apply(lambda x: x.split("/")[0] if "<*sf>" not in x else x.split("/")[1].split("<")[0])

    # print(df["skill_id"])

    df["item_id"] = df["learning_language"] + ":" + df["lexeme_string"]  # .apply(lambda x: x.split("/")[0] if "<*sf>" not in x else x.split("/")[1].split("<")[0])
    df["correct"] = df["session_correct"] / df["session_seen"]
    if not options.continuous_correct:
        df["correct"] = df["correct"].apply(round).astype(int)

    df["combined_language"] = df["ui_language"] + "->" + df["learning_language"]
    df["delta"] /= 60*60

    df = df[['user_id', 'item_id', 'skill_id', 'correct', 'timestamp', 'combined_language', 'history_seen', 'history_correct', 'delta', "session_correct", "session_seen", "learning_language"]]

    if drop_duplicates:
        df.drop_duplicates(subset=["user_id", "item_id", "timestamp"], inplace=True)

    if remove_nan_skills:
        df = df[~df["skill_id"].isnull()]
    else:
        df.loc[df["skill_id"].isnull(), "skill_id"] = "NaN"

    # Create list of KCs
    listOfKC = []
    for kc_raw in df["skill_id"].unique():
        for elt in kc_raw.split('~~'):
            listOfKC.append(elt)
    listOfKC = np.unique(listOfKC)
    print("number of skills:", len(listOfKC), flush=True)

    dict1_kc = {}
    dict2_kc = {}
    for k, v in enumerate(listOfKC):
        dict1_kc[v] = k
        dict2_kc[k] = v

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    unique_items, df["item_id"] = np.unique(df["item_id"], return_inverse=True)

    unique_lang_ids, df["combined_lang_id"] = np.unique(df["combined_language"], return_inverse=True)

    print(list(enumerate(unique_lang_ids)), flush=True)

    df.reset_index(inplace=True, drop=True)  # Add unique identifier of the row
    df["inter_id"] = df.index

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(listOfKC)))
    item_skill = np.array(df[["item_id", "skill_id"]])
    for i in range(len(item_skill)):
        splitted_kc = item_skill[i, 1].split('~~')
        for kc in splitted_kc:
            Q_mat[item_skill[i, 0], dict1_kc[kc]] = 1

    df = df[['user_id', 'item_id', 'timestamp', 'correct', "inter_id", "combined_lang_id", 'history_seen', 'history_correct', 'delta', "session_correct", "session_seen", "learning_language"]]

    if not options.continuous_correct:
        df = df[df.correct.isin([0, 1])]  # Remove potential continuous outcomes
        df['correct'] = df['correct'].astype(np.int32)  # Cast outcome as int32

    # Save data
    modifier = ""
    if subword_skills:
        submodifier = ""
        if tags:
            submodifier += "_tags"
        if lemma:
            submodifier += "_lemma"
        modifier = "_subword"+submodifier+"_vocab="+str(options.vocab_size)+"_nbest="+str(options.nbest)
    elif tags:
        modifier = "_tags"
        if lemma:
            modifier += "_lemma"
    elif lemma:
        modifier = "_lemma"

    if options.continuous_correct:
        modifier += "_continuous"

    sparse.save_npz("E:/project/src/das3h/data/duolingo_hlr/q_mat"+modifier+".npz", sparse.csr_matrix(Q_mat))
    df.to_csv("E:/project/src/das3h/data/duolingo_hlr/preprocessed_data"+modifier+".csv", sep="\t", index=False)

    with open("E:/project/src/das3h/data/duolingo_hlr/skill_map"+modifier+".csv", "w", encoding='utf-8') as skill_map_file:
        for k, v in enumerate(listOfKC):
            skill_map_file.write(str(k) + "\t" + v + "\n")

    with open("E:/project/src/das3h/data/duolingo_hlr/item_map"+modifier+".csv", "w", encoding='utf-8') as item_map_file:
        for k, v in enumerate(unique_items):
            item_map_file.write(str(k) + "\t" + v + "\n")

    return df, Q_mat



def prepare_tagetomo(min_interactions_per_user, remove_nan_skills, subword_skills=False, tags=False, lemma=False, drop_duplicates=False):

    df = pd.read_csv("E:\dataset_memory\d2020_tag.csv", encoding="utf-8")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    df["timestamp"] = df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)

    if subword_skills:
        df["skill_id"] = extract_subword_skills_tage(df["word"], nbest=options.nbest, include_lemma=lemma, tags=tags)
    elif tags:
        df["skill_id"] = extract_tag_skills_tage(df["tag"],  lemma=lemma)
    elif lemma:
        df["skill_id"] = df["tag"].apply(lambda x: x.split("/")[1].split("<")[0])
    else:
        df["skill_id"] = df["word"]

    df["item_id"] = df["format"].apply(str) + ":" + df["tag"]

    df["delta"] /= 60*60

    df = df[['user_id', 'item_id', 'skill_id', 'format', 'correct', 'timestamp', 'history_seen', 'history_correct', 'delta']]

    if drop_duplicates:
        df.drop_duplicates(subset=["user_id", "item_id", "timestamp"], inplace=True)

    if remove_nan_skills:
        df = df[~df["skill_id"].isnull()]

    else:
        df.loc[df["skill_id"].isnull(), "skill_id"] = "NaN"

    # Create list of KCs
    listOfKC = []
    for kc_raw in df["skill_id"].unique():
        for elt in kc_raw.split('~~'):
            listOfKC.append(elt)
    listOfKC = np.unique(listOfKC)
    print("number of skills:", len(listOfKC), flush=True)

    dict1_kc = {}
    dict2_kc = {}
    for k, v in enumerate(listOfKC):
        dict1_kc[v] = k
        dict2_kc[k] = v

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    unique_items, df["item_id"] = np.unique(df["item_id"], return_inverse=True)

    df.reset_index(inplace=True, drop=True)  # Add unique identifier of the row
    df["inter_id"] = df.index

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(listOfKC)))
    item_skill = np.array(df[["item_id", "skill_id"]])
    for i in range(len(item_skill)):
        splitted_kc = item_skill[i, 1].split('~~')
        for kc in splitted_kc:
            Q_mat[item_skill[i, 0], dict1_kc[kc]] = 1

    df = df[['user_id', 'item_id',  'timestamp', 'correct', "inter_id",  'history_seen', 'history_correct', 'delta']]


    # Save data
    modifier = ""
    if subword_skills:
        submodifier = ""
        if tags:
            submodifier += "_tags"
        if lemma:
            submodifier += "_lemma"
        modifier = "_subword"+submodifier+"_vocab="+str(options.vocab_size)+"_nbest="+str(options.nbest)
    elif tags:
        modifier = "_tags"
        if lemma:
            modifier += "_lemma"
    elif lemma:
        modifier = "_lemma"

    sparse.save_npz("E:/project/src/das3h/data/tagetomo/q_mat"+modifier+".npz", sparse.csr_matrix(Q_mat))
    df.to_csv("E:/project/src/das3h/data/tagetomo/preprocessed_data"+modifier+".csv", sep="\t", index=False)

    with open("E:/project/src/das3h/data/tagetomo/skill_map"+modifier+".csv", "w", encoding='utf-8') as skill_map_file:
        for k, v in enumerate(listOfKC):
            skill_map_file.write(str(k) + "\t" + v + "\n")

    with open("E:/project/src/das3h/data/tagetomo/item_map"+modifier+".csv", "w", encoding='utf-8') as item_map_file:
        for k, v in enumerate(unique_items):
            item_map_file.write(str(k) + "\t" + v + "\n")

    return df, Q_mat





if __name__ == "__main__":
    if options.dataset == "tagetomo":
        df, Q_mat = prepare_tagetomo(min_interactions_per_user=options.min_interactions,
                                         remove_nan_skills=options.remove_nan_skills,
                                         tags=options.tags,
                                         lemma=options.lemma,
                                         subword_skills=options.subword_skills)

    elif options.dataset == "duolingo_hlr":
        df, Q_mat = prepare_duolingo_hlr(min_interactions_per_user=options.min_interactions,
                                         remove_nan_skills=options.remove_nan_skills,
                                         tags=options.tags,
                                         lemma=options.lemma,
                                         subword_skills=options.subword_skills)
