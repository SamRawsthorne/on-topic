
import pyLDAvis
from lda_eval_lib.util import create_corpus, save_lda_run, load_lda_run
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import gensim
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os

def visualize_topics(savepath):

    lda_model, textbeforetokenization, tokenized_text, corpus, id2word = load_lda_run(savepath)
    
    
    vocab = list(dict(id2word).values())
    
    # list of document length
    doc_lengths = []
    for k in range(len(corpus)):
        doc_len = 0
        for i,j in corpus[k]:
            doc_len += j
        doc_lengths.append(doc_len)
    
    
    # apply the countvectorizer to get Term frequency
    vectorizer = CountVectorizer(vocabulary = vocab)
    result = vectorizer.transform(textbeforetokenization)
    termfreq = result.sum(axis=0)
    termfreq = np.array(termfreq).reshape(-1)
    

    # model_topics
    modeltopic = lda_model.show_topics(formatted=False, num_words=10, num_topics=-1)    
    topics2keep = []
    for topicidx, words in modeltopic:
        probsum = 0
        for word, prob in words:
            probsum += prob
        if probsum >= 0.01:
            topics2keep.append(topicidx)
    
    # doc_topic matrix excluding insiginificant topics
    model_type = os.path.basename(savepath).split('_')[0]
    if model_type == 'gensim':
        doc_lda = lda_model.get_document_topics(corpus, minimum_probability=0)
        all_topics_csr = gensim.matutils.corpus2csc(doc_lda)
        doc_lda = all_topics_csr.T.toarray()
        new_doc = pd.DataFrame(doc_lda)
        new_doc = new_doc.fillna(0)
        new_doc_colsum = new_doc.sum(axis=1)
        new_doc = new_doc.div(new_doc_colsum, axis=0)

    else:
        doc_lda = []
        for doc in lda_model[corpus]:
            doc_lda.append(doc)
        new_doc = pd.DataFrame([[j for i,j in k] for k in doc_lda])    
        new_doc = new_doc.fillna(0)
        new_doc_colsum = new_doc.sum(axis=1)
        new_doc = new_doc.div(new_doc_colsum, axis=0)        

    doc_top = new_doc.iloc[:,topics2keep]
    doc_top_colsum = doc_top.sum(axis=1)
    doc_top = doc_top.div(doc_top_colsum, axis=0)    

    #topic_term matrix excluding insignificant topics
    topic_term = lda_model.get_topics()
    topic_term = pd.DataFrame(topic_term)
    topic_term = topic_term.loc[topics2keep]
    
    pyldavisdata = {'topic_term_dists': np.array(topic_term),
            'doc_topic_dists': np.array(doc_top),
            'doc_lengths': doc_lengths,
            'vocab': vocab,
            'term_frequency': termfreq}
    return pyldavisdata
