# Comparing the LDA Implementations of Gensim, Mallet, and Tomotopy
# Dr. Zachary Stine
# Jake Davis
# Make sure to tailor the corpus and mallet installation paths to your computer below.
# Custom paths will be in lines 53, 98, 112, 121, and 139.

import nltk
nltk.download("stopwords")

# python library imports
import numpy as np
import json
import glob
import logging

# gensim imports
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import datapath

# mallet imports
import os
import little_mallet_wrapper as lmw

# tomotopy imports
import tomotopy as tp

# spacy
import spacy
from nltk.corpus import stopwords

# vis (May use for visualization)
# import pyLDAvis
# import pyLDAvis.gensim_models

# May replace with original stopwords at some point
stopwords = stopwords.words("english")
# stopwords.extend(['example', 'words'])

# Preparing the Data-----------------------------------------------------------
def load_data(file):
    with open (file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data)

def write_data(file, data):
    with open (file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# import the designated corpus here.
data = load_data("C:/Users/mercu/Downloads/ushmm_dn.json")["texts"]

def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return (final)

def remove_stopwords(texts):
   return [[word for word in simple_preprocess(str(doc)) 
   if word not in stopwords] for doc in texts]

data_words = gen_words(data)
data_words_nostops = remove_stopwords(data_words)
print (data_words_nostops[0][0:20])

id2word = corpora.Dictionary(data_words_nostops)
texts = data_words_nostops
corpus = [id2word.doc2bow(text) for text in texts]
# print(corpus[:4]) #it will print the corpus we created above.

# [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:4]] 
# it will print the words with their frequencies.

# Begin 3 Different LDA Implementations-------------------------------------

# Figure out loop logic for list of k (topic) values
# Figure out multiple models of each implementation, e.g. input number of iterations 
# and assume that for each
# Figure out file structure for outputs of said models, e.g. parent folder k-20 and 
# gensim_k-20_1, _2, mallet_k-20_1, 2, then k-100 so on

# Gensim Implementation-----------------------------------------------------
gensim_lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=20,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=5,
                                            alpha="auto")

# Save Gensim model to disk
# Make sure to put your designated file path here.
gensim_file = datapath("C:/Users/mercu/Downloads/gensim_result")

gensim_lda_model.save(gensim_file)

# Load Gensim model from disk (optional)

# from gensim import  models

# loaded_gensim_lda = models.ldamodel.LdaModel.load(gensim_file)


# Mallet Implementation----------------------------------------------------
# Make sure to set the mallet_path to your computer's installation.
os.environ.update({'MALLET_HOME':r'C:/mallet-2.0.8/'})
mallet_path = "C:/mallet-2.0.8/bin/mallet"

mallet_lda_model = gensim.models.wrappers.LdaMallet(mallet_path, 
                                                    corpus=corpus, 
                                                    num_topics=20, 
                                                    id2word=id2word)

# Save Mallet model to disk
# Make sure to put your designated file path here.
mallet_file = datapath("C:/Users/mercu/Downloads/mallet_result")

mallet_lda_model.save(mallet_file)

# Tomotopy Implementation--------------------------------------------------
tomotopy_lda_model = tp.LDAModel(k=20)
for vec in texts: 
    tomotopy_lda_model.add_doc(vec)

# Research burn in method more, not fully sure what it does.
tomotopy_lda_model.burn_in = 100

iter=50
for i in range(0, iter, 5):
    tomotopy_lda_model.train(50)

# Save Tomotopy model to disk
# Make sure to put your designated file path here.
tomotopy_file = 'C:/Users/mercu/Downloads/tomotopy_result'

tomotopy_lda_model.save(tomotopy_file, True)
