import fasttext.util
import numpy as np


# fasttext.util.download_model('en',  if_exists='ignore')

en = fasttext.load_model('cc.en.300.bin')

data = "E:\dataset_memory\data2020-4-6.csv"
words = []
# ['de', 'es', 'pt', 'en', 'fr', 'it']
model = en

print('embedding...')

with open(data, encoding="utf-8") as data_file:
    data_file.readline()
    for i, line in enumerate(data_file):
        _, _, _,  word, _, _, _, _, _, _, _, _ = line.strip().split(",")
        embedding = model[word]
        embedding = [float(str(item).replace('\n', '')) for item in embedding.tolist()]

        reg = word + "\t" + 'en' + "\t" + str(embedding)
        if reg not in words:
            words.append([reg])
        if i%100000 == 0:
            print(i,'/25309814', flush=True)

print('saving...')

with open('all_word_embeddings_fastword.csv', 'w') as csvfile:
    for i in range(len(words)+1):
        csvfile.write(words[i])
        csvfile.write('\n')





