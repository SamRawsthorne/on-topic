
import numpy as np
import os
import gensim 
from lda_eval_lib.util import create_corpus, save_lda_run, load_lda_run
import lda_eval_lib.gensim_ldamallet as mallet
from gensim.models import CoherenceModel
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer



def IDXtopics2keep(modelpath):
    """
    modelpath = path to the folder where your model is located

    Function to identify null topics that feature a uniform word distribution (top 10 words and bottom 10 words have the same probabilities)
    
    Output: the number of valid topics (Scalar)
    """
    lda_model, textbeforetokenization, tokenized_text, corpus, id2word = load_lda_run(modelpath)
    TopicWordDist = lda_model.get_topics()
    TopicWordDist = pd.DataFrame(TopicWordDist)

    TopicWordDist = np.array(TopicWordDist)
    topics2keep = []
    for i in range(len(TopicWordDist)):
        arr = TopicWordDist[i,:]    
        arr = np.squeeze(np.array(arr))
        indices =  np.argpartition(arr, -10)[-10:]
        # Get the largest 10 values using the indices
        largest_10_values = arr[indices]
        
        # Sort the largest 10 values in descending order
        largest_10_values = np.sort(largest_10_values)[::-1]
        
        indices = np.argpartition(arr, 10)[:10]
        # Get the smallest 10 values using the indices
        bottom_10_values = arr[indices]
        
        # Optionally, sort the bottom 10 values
        bottom_10_values = np.sort(bottom_10_values)
        diff = sum(largest_10_values) - sum(bottom_10_values)
        if diff>=0.01:
            topics2keep.append(i)
    return len(topics2keep)/len(TopicWordDist)

# function to evaluate the diversity of topic keywords
def diversity(modelpath):
    """

    output: uniqueness of top topic words (scalar) 
    
    """
    
    lda_model, textbeforetokenization, tokenized_text, corpus, id2word = load_lda_run(modelpath)

    topics_lda = lda_model.show_topics(formatted=False, num_words=10, num_topics=-1)
    # get top 10 words in the format [['fruit','apple','banana'],['cosine','sine','tangent'],['apple','mobile','ditigal']]
    topic_words = [[keyword for keyword, _ in sublist] for _, sublist in topics_lda]
    topic_words = [words for words in topic_words if len(words) > 0]
    topic_words = [[element for element in sublist if element != ''] for sublist in topic_words]

    unique_words = set()
    for topic in topic_words:
        unique_words = unique_words.union(set(topic))
    allwords = 0
    for topic in topic_words:
        allwords += len(topic)
    td = len(unique_words) / allwords
    return td


def granularity(modelpath):
    """
    1 - average document frequency of topic keywords
    
    output: granularity of top topic words (scalar) 
    
    """
    # generate a dictionary of topic top N keywords 

    lda_model, textbeforetokenization, tokenized_text, corpus, id2word = load_lda_run(modelpath)
    
    topics_lda = lda_model.show_topics(formatted=False, num_words=10, num_topics=-1)
    # get top 10 words in the format [['fruit','apple','banana'],['cosine','sine','tangent'],['apple','mobile','ditigal']]
    topic_words = [[keyword for keyword, _ in sublist] for _, sublist in topics_lda]
    topic_words = [words for words in topic_words if len(words) > 0]
    topic_words = [[element for element in sublist if element != ''] for sublist in topic_words]

    vocabid = 0
    topickeyworddict = {}
    for topic in topic_words:
        for word in topic:
            if word not in topickeyworddict.keys():
                topickeyworddict[word]=vocabid
                vocabid+=1
    # Use the dictionary to create a binary countvectorizer (to count document frequency of terms)
    vectorizer = CountVectorizer(vocabulary = topickeyworddict, binary=True)
    
    # apply the countvectorizer
    result = vectorizer.transform(textbeforetokenization)
    
    # Get indices of topic keywords
    topic_keywords = [[topickeyworddict[keyword] for keyword in topic] for topic in topic_words]

    # a list to store average DF of keywords within each topic.
    granularity_scores = []
    for i in range(len(topic_keywords)):
        doc_frequencies = result[:,topic_keywords[i]].sum(axis=0) # get document frequency of topic keywords
        average_doc_frequency = np.mean(doc_frequencies) # calculate average document frequency of the topic
        granularity_scores.append(average_doc_frequency) 
        
    ndocs = len(textbeforetokenization) # number of documents

    average_granularity = 1 - np.mean(granularity_scores)/ndocs # calculate average DF across all topics and divide it by ndocs for normalisation. 
    return average_granularity

def perplexity(modelpath):
    lda_model, textbeforetokenization, tokenized_text, corpus, id2word = load_lda_run(modelpath)
    
    topics_lda = lda_model.show_topics(formatted=False, num_words=10, num_topics=-1)
    # get top 10 words in the format [['fruit','apple','banana'],['cosine','sine','tangent'],['apple','mobile','ditigal']]
    topic_words = [[keyword for keyword, _ in sublist] for _, sublist in topics_lda]
    topic_words = [words for words in topic_words if len(words) > 0]
    topic_words = [[element for element in sublist if element != ''] for sublist in topic_words]

    model_type = os.path.basename(modelpath).split('_')[0]
    if model_type == 'mallet':
        convertedlda = mallet.malletmodel2ldamodel(lda_model)
        perplexity = convertedlda.log_perplexity(corpus)
    else:
        perplexity = lda_model.log_perplexity(corpus)
    return perplexity

def coherence(modelpath):
    lda_model, textbeforetokenization, tokenized_text, corpus, id2word = load_lda_run(modelpath)
    
    topics_lda = lda_model.show_topics(formatted=False, num_words=10, num_topics=-1)
    # get top 10 words in the format [['fruit','apple','banana'],['cosine','sine','tangent'],['apple','mobile','ditigal']]
    topic_words = [[keyword for keyword, _ in sublist] for _, sublist in topics_lda]
    topic_words = [words for words in topic_words if len(words) > 0]
    topic_words = [[element for element in sublist if element != ''] for sublist in topic_words]
    coherence_model_lda = CoherenceModel(topics=topic_words,
                                texts=tokenized_text,
                                corpus=corpus,
                                dictionary=id2word,
                                coherence='c_v')
    
    coherence = coherence_model_lda.get_coherence()    
    return coherence
