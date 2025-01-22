from lda_eval_lib.custom_mallet import CustomLdaMallet
from lda_eval_lib.util import create_corpus, save_lda_run, load_lda_run
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import os
import time 
import pickle

def lda_run(tokenized_text, minpc, maxpc, alpha, beta, ntopics, mallet_path, savefolder, gensimpasses = 100, gensimiterations = 100, malletiterations = 2000 ):
    
    textbeforetokenization, id2word, corpus, wordcounts = create_corpus(tokenized_text, minpc, maxpc)

    model_name = "gensim_"+str(minpc)+"_"+str(maxpc)+"_"+str(ntopics)+"_"+str(alpha)+"_"+str(beta)
    
    start = time.time()
    print(' - modelling ', model_name)
        
    lda_model = LdaModel(corpus=corpus,
                            id2word=id2word,
                            num_topics=ntopics,
                            update_every=1,
                            chunksize=100,
                            passes=gensimpasses,
                            iterations = gensimiterations,
                            alpha=alpha,
                            eta=beta,
                            per_word_topics=True,
                            minimum_probability=0.0
                           )

    save_lda_run(savefolder, model_name, lda_model, textbeforetokenization, tokenized_text, corpus, id2word)
    print(f'model saved: {os.path.join(savefolder,model_name)}')

    #### MALLET ####
    model_name_mallet = "mallet_"+str(minpc)+"_"+str(maxpc)+"_"+str(ntopics)+"_"+str(alpha)+"_"+str(beta) 

    print(' - modelling ', model_name_mallet)
    start = time.time()

    mallet_model = CustomLdaMallet(
        mallet_path=mallet_path,
        corpus=corpus,
        id2word=id2word,
        num_topics=ntopics,
        alpha=alpha,
        beta=beta,
        iterations=malletiterations
    )
    
    save_lda_run(savefolder, model_name_mallet, mallet_model, textbeforetokenization, tokenized_text, corpus, id2word)
    print(f'model saved: {os.path.join(savefolder,model_name_mallet)}')
