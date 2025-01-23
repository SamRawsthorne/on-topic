
import pandas as pd
import numpy as np
import pickle
from lda_eval_lib.util import create_corpus, save_lda_run, load_lda_run
import random
import openai
from openai import OpenAI
import os
import getpass
import json

# Function to generate the top n words in a topic. If n is an integer, then we return the n words. If n is less
# than one, then n is the percentile over which we capture. If ascending is False then we capture the most
# common words. Otherwise, we capture the least common words. If remove_zeroes is true, then we focus only on
# words with non-zero weighting. This may be helpful if there are many words with zero weighting in a topic and
# we want the n least common words to the topic. (Otherwise, only words from the end of the alphabet are
# returned. Or only very sparse words which do not feature as high-intensity words in any topic)
def get_n_words(phi, n, ascending=False, remove_zeroes=False):
    if n < 1:
        n = int(round(n * len(phi.index), 0))
    # get top n words across all topics
    top_n = dict()
    if ascending:
        if remove_zeroes:
            for j in phi.columns:
                temp = phi.loc[phi[j] != 0]
                top_n[j] = list(temp[j].sort_values(ascending=True).index)[:n]
        else:
            for j in phi.columns:
                top_n[j] = list(phi[j].sort_values(ascending=True).index)[:n]
    else:
        for j in phi.columns:
            top_n[j] = list(phi[j].sort_values(ascending=False).index)[:n]
    return top_n


# Function to shuffle the words and intruder word. DF contains the words and intruder words. word_cols is the
# columns containing the words and the intruder word.
def shuffle_words(df, word_cols):
    word_intrusion_list = []
    for a in range(len(df.index)):
        words = list(df.loc[a, word_cols])
        random.shuffle(words)
        word_intrusion_list.append(', '.join(words))
    return word_intrusion_list


# Function to calculate the number of chunks to ensure GPT completes all topics
def chunk_word_intrusion_list(word_intrusion_list, chunk_size=10):
    if len(word_intrusion_list) % chunk_size != 0:
        chunks = int(len(word_intrusion_list) / chunk_size) + 1
    else:
        chunks = int(len(word_intrusion_list) / chunk_size)
    return chunks


# Function to generate system message which introduces the task
def generate_system_message():
    # System message
    system_message = """
    In this research project, I have constructed a corpus from UK annual reports where firms describe their business models, strategies, risks and uncertainties, outlook, financial and non-financial results, and corporate governance. 

    A topic modelling technique was applied to this corpus. As a research associate, your task is to evaluate the quality of the generated topics. You will be provided with a series of lists, each containing six keywords. Among these keywords, one is an intruder word that does not belong to the topic.

    Identify the intruder word that is least likely to appear with the other keywords in the list. Note that 'ps' stands for pound sterling, 'bn' stands for billion, and 'fy' stands for fiscal year.

    Examples of input and output

    Input: "Topic 131:['lending', 'funding', 'investors', 'uneconomic', 'songwriters', 'borrowers']" 
    Output: {'songwriters'}

    Input: "Topic 132:['principal', 'risks', 'aircraft', 'combination', 'uncertainties', 'continued']"
    Output: {'aircraft'}

    Input: "Topic 133:['gifts', 'collateral', 'gift', 'entertainment', 'register', 'hospitality']"
    Output: {'register'}

    Provide your answer in JSON format.

    Example of the output format: {'Topic 131':'songwriters' ,'Topic 132':'aircraft', 'Topic 133':'register'}
    """
    return system_message


# Function to perform the WIT for a given list of topics where each topic is a string of words separated by a comma
def perform_task(list_of_topics, chunk_size, iteration, gpt_model, temperature, client):
    system_message = generate_system_message()

    user_message = ""
    for t in range(len(list_of_topics)):
        # generate user message for the chunk
        user_message = user_message + f" Topic {iteration * chunk_size + t + 1}:{list_of_topics[t]} \n"

    # create a ChatGPT completion session
    completion = client.chat.completions.create(
        model=gpt_model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )

    # get ChatGPT's response and store the response to the list 'intruderbyGPT'
    result = json.loads(dict(completion.choices[0].message)['content'])
    return result


# Defining exception class in case no valid intruder word can be found
class SetLengthWarning(UserWarning):
    pass

# Function to identify error
def check_valid_intruder(set):
    if len(set) == 0:
        warnings.warn(f"No valid intruder words are found; consider changing thresholds! Skipping topic...",
                      SetLengthWarning)
        return 'Invalid'
    else:
        return 'Valid'


def OpenAI_client():
    if 'OPENAI_API_KEY' not in os.environ:
        os.environ['OPENAI_API_KEY'] = getpass.getpass(prompt='Enter your API key: ')
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    return client

# Main function to complete the analysis
def word_intrusion_task(savepath, client, top_n, bottom_n, common_n, chunk_size=10,
                           gpt_model="gpt-4o", temperature=0):
    # Get the topic-word matrix
    model_name = os.path.basename(savepath)
    model, textbeforetokenization, tokenized_text, corpus, id2word = load_lda_run(savepath)

    phi = model.get_topics()
    phi = pd.DataFrame(phi)
    phi = phi.rename(columns=id2word.id2token)
    phi = phi.T

    # Retrieve the top 5 words for each topic
    top_n_words = get_n_words(phi, top_n, False)

    # To construct the intruder word, we need the 15% least probable words for the given topic that also appear in
    # the top 20 most common words in at least one other topic.
    bottom_n_words = get_n_words(phi, bottom_n, True, True)
    common_n_words = get_n_words(phi, common_n, False)
    all_common_n_words = list(set([item for sublist in common_n_words.values() for item in sublist]))

    word_intrusion_task = []
    for j in range(len(top_n_words)):
        # Collect the top n words in the topic
        top_words = top_n_words[j]
        # Remove from the set of common words across the T topics those which appear commonly in the current topic
        sub_common_words = set(all_common_n_words) - set(common_n_words[j])
        # Retain only those which appear in the bottom X(%) of words in the topic
        sub_common_words = sub_common_words & set(bottom_n_words[j])
        # Check whether there is a valid set of candidate intruder words (in case the thresholds are too restrictive)
        check = check_valid_intruder(sub_common_words)
        # Choosing a random word as the intruder if set of candidate intruder words is valid
        if check == 'Valid':
            intruder = random.choice(list(sub_common_words))
        else:
            intruder = 'INVALID_INTRUDER'
        word_intrusion_task = word_intrusion_task + [[model_name, j] + top_words + [intruder]]
    word_intrusion_task_df = pd.DataFrame(word_intrusion_task)
    word_intrusion_task_df.columns = ['Model', 'Topic'] + [f'Word{j}' for j in range(top_n)] + ['Intruder']

    # Shuffling words into random order
    word_intrusion_list = shuffle_words(word_intrusion_task_df, [f'Word{j}' for j in range(top_n)] + ['Intruder'])

    # Calculating the number of chunks to ensure GPT completes all topics
    chunks = chunk_word_intrusion_list(word_intrusion_list, chunk_size)

    # List to store intruder word identified by GPT
    intruderbyGPT = []

    # system message

    # Apply GPT
    for c in range(chunks):
        chunk = word_intrusion_list[c * chunk_size:c * chunk_size + chunk_size]
        result = perform_task(chunk, chunk_size, c, gpt_model, temperature, client)
        intruderbyGPT = intruderbyGPT + list(result.values())

    # Save the response to a new column
    word_intrusion_task_df['Response'] = intruderbyGPT

    # Calculate the accuracy of intrusion (after removing any invalid intruders)
    temp = word_intrusion_task_df[word_intrusion_task_df['Intruder'] != 'INVALID_INTRUDER']
    acc = temp[temp['Response'] == temp['Intruder']]
    acc = len(acc) / len(temp.index)

    # Storing macro data
    macro_data = pd.DataFrame({'model_name': [model_name], 'task': [word_intrusion_task_df], 'accuracy': [acc]})

    return macro_data

