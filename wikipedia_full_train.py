#!/usr/bin/env python
# -*- coding: utf-8 -*-
datadump_dir = "/home/rohit.katariya/rohit/sentence_classification/datadump/"
#from gensim.corpora import WikiCorpus, MmCorpus
#print("starting WikiCorpus command")
#wiki = WikiCorpus(datadump_dir+"enwiki-latest-pages-articles.xml.bz2" , processes=20) 
#print("starting serialize command")
#MmCorpus.serialize(datadump_dir+'wiki_en_vocab200k.mm', wiki)



import logging
import os.path
import sys
import multiprocessing

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    inp=datadump_dir+"enwiki-latest-pages-articles.xml.bz2"
    outp1= datadump_dir+"wiki_full_model.text.model"
    outp2 = datadump_dir +  "wiki_full.text.vector"

    model = Word2Vec(LineSentence(inp), size=50, window=5, min_count=3,
            workers=20)

    # trim unneeded model memory = use(much) less RAM
    #model.init_sims(replace=True)
    model.save(outp1)
    model.save_word2vec_format(outp2, binary=False)
