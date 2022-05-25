import numpy as np
import pandas as pd

import fasttext

EMBEDDING_SIZE = 300

all_words = []
ft_by_lang = {}
langs = set()

with open("duolingo_en.csv", encoding="utf-8") as words_file:
    for i, line in enumerate(words_file):
        if i>0:
            _,_, _, _, _, lang, _, _, lexeme_string,_, _, _, _, = line.strip().split(",")
            word = lexeme_string.split("/")[0]
            if word == "<*sf>":
                word =lexeme_string.split("/")[1].split("<")[0]
            langs.add(lang)
            all_words.append((word, lang))

print("loaded all words...")

for lang in langs:
    ft_by_lang[lang] = fasttext.load_model('cc.' + lang + ".300.bin")
    print("loaded", lang, "model")

all_words = sorted(list(set(all_words)))
res_save = []
res_save.append(np.zeros(300)) # padding 0
with open("en_word_embeddings_fastword.csv", "w", encoding="utf-8") as out_file:
    for i, (word, lang) in enumerate(all_words):
        embedding = ft_by_lang[lang][word]

        assert len(embedding) == EMBEDDING_SIZE

        res = word + "\t" + lang + "\t" + str(list(embedding)) + "\n"
        emb = list(embedding)
        res_save.append(emb)
        out_file.write(res)
        print(i, "done")

np.save("en_word_embeddings_fastword.npy",np.array(res_save))