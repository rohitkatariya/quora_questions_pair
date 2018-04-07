#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from gensim import corpora
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity
from six import iteritems
import sys
sys.path.append("./../")
from libraries.csvFunctions import readCsvToDictList
import nltk
from nltk.corpus import stopwords
import traceback
def get_questions(input_corpus_file):
    read_csv = readCsvToDictList(input_corpus_file)
    questions_all = set([])
    for line in read_csv:
        questions_all.add(line["question1"])
        questions_all.add(line["question2"])
    return questions_all
input_corpus_file = 'datadump/lsi_data/train.csv'
all_questions = get_questions(input_corpus_file)



stoplist  = set(stopwords.words('english'))
all_questions_tokenized = []
for line in all_questions:
    if not line:
        continue
    try:
        #tokenized_sentence = line.lower().split() 
        tokenized_sentence = nltk.word_tokenize( line.lower())
        all_questions_tokenized.append(tokenized_sentence)
    except:
        print( line)
        traceback.print_exc()
#         raw_input("K?") 
dictionary = corpora.Dictionary(all_questions_tokenized)
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq <=5]
dictionary.filter_tokens(stop_ids + once_ids)
dictionary.compactify()
print(dictionary)
class MyCorpus(object):
    def __iter__(self):
        for line in all_questions:
            if line:
                yield dictionary.doc2bow(line.lower().split())
 
corpus_memory_friendly = MyCorpus()
i=0
# for vector in corpus_memory_friendly:  # load one vector into memory at a time
#     print(vector)
#     i+=1
#     if i==40:
#         break
lsi = LsiModel(corpus_memory_friendly, id2word=dictionary, num_topics=300)
corpora.MmCorpus.serialize('datadump/lsi_data/train_corpora.mm', corpus_memory_friendly)
dictionary.save('datadump/lsi_data/train.dict')
lsi.save("datadump/lsi_data/lsi_model")  
# dictionary = corpora.Dictionary(corpus)
# corpus_gensim = [dictionary.doc2bow(doc) for doc in corpus]
# tfidf = TfidfModel(corpus_gensim)
# corpus_tfidf = tfidf[corpus_gensim]
# lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)
# lsi_index = MatrixSimilarity(lsi[corpus_tfidf])
# sims['ng20']['LSI'] = np.array([lsi_index[lsi[corpus_tfidf[i]]]
#                                 for i in range(len(corpus))])
