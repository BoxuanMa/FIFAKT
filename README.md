
# Each Encounter Counts: Modeling Language Learning and Forgetting


This is the repository for the code in the paper FIFAKT: Each Encounter Counts: Modeling Language Learning and Forgetting. LAK23: 13th International Learning Analytics and Knowledge Conference. Authors: Boxuan Ma, Gayan Prasad Hettiarachchi, Sora Fukui and Yuji Ando. (to be appear)

If you find this repository useful, please cite our work

```
@inproceedings{ma2023each,
  title={Each Encounter Counts: Modeling Language Learning and Forgetting},
  author={Ma, Boxuan and Hettiarachchi, Gayan Prasad and Fukui, Sora and Ando, Yuji},
  booktitle={LAK23: 13th International Learning Analytics and Knowledge Conference},
  pages={79--88},
  year={2023}
}
```

## Abstract 

Language learning applications usually estimate the learner's language knowledge over time to provide personalized practice content for each learner at the optimal timing. However, accurately predicting language knowledge or linguistic skills is much more challenging than math or science knowledge, as many language tasks involve memorization and retrieval. Learners must memorize a large number of words and meanings, which are prone to be forgotten without practice. Although a few studies consider forgetting when modeling learners' language knowledge, they tend to apply traditional models, consider only partial information about forgetting, and ignore linguistic features that may significantly influence learning and forgetting. This paper focuses on modeling and predicting learners' knowledge by considering their forgetting behavior and linguistic features in language learning. Specifically, we first explore the existence of forgetting behavior and cross-effects in real-world language learning datasets through empirical studies. Based on these, we propose a model for predicting the probability of recalling a word given a learner’s practice history. The model incorporates key information related to forgetting, question formats, and semantic similarities between words using the attention mechanism. Experiments on two real-world datasets show that the proposed model improves performance compared to baselines. Moreover, the results indicate that combining multiple types of forgetting information and item format improves performance. In addition, we find that incorporating semantic features, such as word embeddings, to model similarities between words in a learner's practice history and their effects on memory also improves the model.

## Datasets

The Duolingo data we used can be [downloaded](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME) from Duolingo. We also use their [HLR code](https://github.com/duolingo/halflife-regression) as a baseline for our experiments. 

The Tagetomo data we used is private, which is belong to Obunsha.inc, so we cannot make it public.


## Code

Some of the code in this repository is based on code from [Benoît Choffin](https://github.com/BenoitChoffin)'s [DAS3H repository](https://github.com/BenoitChoffin/das3h) and [Brian Zylich](https://github.com/bzylich)'s [Linguistic Skill Modeling repository](https://github.com/bzylich/linguistic-skill-modeling).





