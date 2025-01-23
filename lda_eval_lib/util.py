import gensim
import gensim.corpora as corpora
import pickle
from openai import OpenAI
import os
import getpass
from lda_eval_lib.gensim_ldamallet import LdaMallet
import lda_eval_lib

def OpenAI_client():
    if 'OPENAI_API_KEY' not in os.environ:
        os.environ['OPENAI_API_KEY'] = getpass.getpass(prompt='Enter your API key: ')
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    return client

def create_corpus(tokenized_text,minpc,maxpc):
    """
    Function to create a corpus and a dictionary for LDA model

    Inputs:
        tokenized_text : tokenised input text in the following format. 
        [['manufacture','firm','supply','primary','power'], ['train','sesseion','iterate','']]

        minpc : filter words with document frequency lower than 'minpc' 

        maxpc : filter words with document frequency greater than 'maxpc'

    Output:
        textbeforetokenization: text before tokenisation. It it used for the calculation of evaluation metrics
        id2word: dictionary
        corpus: text converted into bag-of-words
        wordcounts: word count for each document
    """
    # Text before tokenization
    textbeforetokenization = [' '.join(tokens) for tokens in tokenized_text] 

    # Create Dictionary
    id2word = corpora.Dictionary(tokenized_text)

    # Filtering too frequent/infrequent words
    id2word.filter_extremes(no_below=round(minpc * len(tokenized_text)), no_above= maxpc)

    # Create Corpus (use the created dictionary to create corpus)
    corpus = [id2word.doc2bow(text) for text in tokenized_text]

    wordcounts = []
    for doc in corpus:
        words = [j for i,j in doc]
        wordcounts.append(len(words))

    return textbeforetokenization, id2word, corpus, wordcounts

def save_lda_run(savefolder, model_name, lda_model, textbeforetokenization, tokenized_text, corpus, id2word):
    """
    Save the LDA run's output to a specified path.
    Args:
        output_path (str): Directory to save the files.
        lda_model (LdaModel): Trained LDA model.
        textbeforetokenization (list): Raw text documents before tokenization.
        tokenized_text (list): Tokenized text documents.
        corpus (list): Bag-of-words corpus.
        id2word (Dictionary): Gensim dictionary object.
    """
    # Save the LDA model
    savepath = os.path.join(savefolder,model_name)
    os.makedirs(savepath, exist_ok=True)
    lda_model.save(os.path.join(savepath,"lda_model"))
    
    # Save other components
    with open(os.path.join(savepath,"textbeforetokenization.pkl"), "wb") as f:
        pickle.dump(textbeforetokenization, f)
    with open(os.path.join(savepath,"tokenized_text.pkl"), "wb") as f:
        pickle.dump(tokenized_text, f)
    with open(os.path.join(savepath,"corpus.pkl"), "wb") as f:
        pickle.dump(corpus, f)
    id2word.save(os.path.join(savepath,"id2word"))

def load_lda_run(savepath):
    """
    Load the LDA run's output from a specified path.
    Args:
        savefolder (str):path to the model folder
    Returns:
        lda_model: Trained LDA model.
        textbeforetokenization: Raw text documents before tokenization.
        tokenized_text: Tokenized text documents.
        corpus: Bag-of-words corpus.
        id2word: Gensim dictionary object.
    """
    # Load the LDA model
    #savepath = os.path.join(savefolder,model_name)
    model_type = os.path.basename(savepath).split('_')[0]
    if model_type == 'gensim':
        lda_model = gensim.models.ldamodel.LdaModel.load(os.path.join(savepath,"lda_model"))
    else:
        lda_model = lda_eval_lib.gensim_ldamallet.LdaMallet.load(os.path.join(savepath,"lda_model"))
    
    # Load other components
    with open(os.path.join(savepath,"textbeforetokenization.pkl"), "rb") as f:
        textbeforetokenization = pickle.load(f)
    with open(os.path.join(savepath,"tokenized_text.pkl"), "rb") as f:
        tokenized_text = pickle.load(f)
    with open(os.path.join(savepath,"corpus.pkl"), "rb") as f:
        corpus = pickle.load(f)
    id2word = corpora.Dictionary.load(os.path.join(savepath,"id2word"))
    
    return lda_model, textbeforetokenization, tokenized_text, corpus, id2word


