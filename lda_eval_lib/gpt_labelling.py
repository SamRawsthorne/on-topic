
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
import re

def OpenAI_client():
    if 'OPENAI_API_KEY' not in os.environ:
        os.environ['OPENAI_API_KEY'] = getpass.getpass(prompt='Enter your API key: ')
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    return client

# Function to calculate the number of chunks to ensure GPT completes all topics
def chunk_topics(topic_df, chunk_size=10):
    if len(topic_df.columns) % chunk_size != 0:
        chunks = int(len(topic_df.columns) / chunk_size) + 1
    else:
        chunks = int(len(topic_df.columns) / chunk_size)
    return chunks


# Function to generate the task description
def generate_task_description(type, nwords, ntopics):
    if type == 'naive':
        task_description = """
                            You are provided with the results of a topic modeling analysis. 
                            You are provided with the index of the topic (e.g. Topic1) and top {nwords} words for the topic. 
                            Please provide a label which best describes the topic and the rationale for the label.
                            Return only the label and rationale for the topic.
                            Do not use quotation marks in the label or rationale. 
                            Provide Your label and rationale as a string separated by a semicolon like in the following example:
                            [Label; Rationale]
                            """
        task_description = task_description.format(
            **{"nwords": nwords})
    else:
        json_output = """{topics: { 
                                   "topic": "Topic0",
                                   "label" : "Label0",
                                   "rationale": "Rationale0"
                                    }, 
                                    {
                                    "topic": "Topic1",
                                    "label" : "Label1",
                                    "rationale": "Rationale1"
                                    }
                            }"""
        if type == 'simple':
            task_description = """
                            You are provided with the results of a topic modeling analysis. You are provided with the top {nwords} words for each of the {ntopics} topics. 
                            Return only the labels and rationales for each topic. 
                            Provide Your labels and rationales in JSON format like in the following example:
                            {json_output}
                            """
            task_description = task_description.format(
                **{"nwords": nwords, "ntopics": ntopics, "json_output": json_output})
        elif type == 'context':
            task_description = """
                        For context, in a research project, a corpus of 'Principal Risks and Uncertainties' sections from UK annual reports has been created. 
                        An LDA topic model has then been constructed based on the text. The analysis generated {ntopics} topics.
                        You are a research associate who is tasked with interpreting the output of topic models generated from corporate disclosures.
                        Your objective is to provide a label which best represents the semantic meaning of the topic. 
                        You are provided with the top {nwords} words for each of the {ntopics} topics. 
                        Return only the labels and rationales for each topic. 
                        Provide Your labels and rationales in JSON format like in the following example:
                        {json_output}                        
                        """
            task_description = task_description.format(
                **{"nwords": nwords, "ntopics": ntopics, "json_output": json_output})
        elif type == 'toc':
            task_description = """
                                For context, in a research project, a corpus of 'Principal Risks and Uncertainties' sections from UK annual reports has been created. 
                                An LDA topic model has then been constructed based on the text. The analysis generated {ntopics} topics.
                                You are a research associate who is tasked with interpreting the output of topic models generated from corporate disclosures.
                                Your objective is to provide a label which best represents the semantic meaning of the topic.
                                Your goal is to review these keywords, generate specific labels for each topic, and ensure that the labels are mutually exclusive.
                                You are provided with the top {nwords} words for each of the {ntopics} topics.
                                Instructions:
                                    1.  Read and Analyze Keywords:
                                    1.a Carefully read the list of keywords for each of the 35 topics.
                                    1.b Identify semantic links between the keywords within each topic.
                                    2.  Generate Initial Labels:
                                    2.a Based on your analysis, generate a specific and descriptive label for each topic.
                                    2.b Labels should reflect the specific nature of the risks and uncertainties rather than generic terms like 'risk management' or 'risks.'
                                    3.  Ensure Mutually Exclusive Labels:
                                    3.a Compare topics with similar labels.
                                    3.b Identify subtle differences between the topics.
                                    3.c Modify the labels to ensure each one is unique and mutually exclusive.
                                    Example: 
                                    Input word lists
                                    Topic 1: ['regulation', 'compliance', 'legal', 'policy', 'law', 'rules', 'regulatory', 'governance', 'legislation', 'audit']
                                    Topic 2: ['market', 'competition', 'demand', 'consumer', 'price', 'sales', 'industry', 'growth', 'revenue', 'trend']

                                    Topic 1:
                                    •   Label: Regulatory Compliance Risks
                                    •   Rationale: The keywords are centered around legal and regulatory aspects, indicating risks associated with compliance with laws and regulations. 
                                    Topic 2:
                                    •   Label: Market Competition Risks
                                    •   Rationale: The keywords suggest risks related to market dynamics, competition, and consumer behavior affecting the company's performance.

                                Return only the labels and rationales for each topic. 
                                Provide Your labels and rationales in JSON format like in the following example:
                                {json_output}
                                """
            task_description = task_description.format(
                **{"nwords": nwords, "ntopics": ntopics, "json_output": json_output})
        else:
            task_description = "Error"
    return task_description.strip()


def generate_content(ntopics, df, nwords, topic_name=''):
    if ntopics == 1:
        topics = topic_name + ': ' + ', '.join(list(df.sort_values(t, ascending=False).index)[:nwords])
        content = """
                    Provide your label and rationale for the following keyword list: {topics}
                    """
        content = content.format(
            **{"nwords": nwords, "topics": topics})
    else:
        topics = ''
        for i in list(df.columns):
            topics = topics + i + ': ' + ' '.join(list(df.sort_values(i, ascending=False).index)[:nwords]) + '; '
        content = """
                        Include labels and rationales for all of the the following {ntopics} keyword lists: {topics}
                        """
        content = content.format(
            **{"nwords": nwords, "topics": topics, "ntopics": ntopics})
    return content.strip()


# Function to generate the naive prompt which passes one topic at a time to CGP
def generate_naive_prompt(topic, nwords):
    prompt_template = """
                    You are provided with the results of a topic modeling analysis. 
                    You are provided with the index of the topic (e.g. Topic1) and top {nwords} words for the topic. 
                    Please provide a label which best describes the topic and the rationale for the label.
                    Return only the label and rationale for the topic.
                    Do not use quotation marks in the label or rationale. 
                    Provide Your label and rationale as a string separated by a semicolon like in the following example:
                    [Label; Rationale]
                    Provide your label and rationale for the following keyword list: {topic}
                    """
    prompt = prompt_template.format(
        **{"nwords": nwords, "topic": topic})
    return prompt


# Function to execute the naive prompt
def get_labels_naive(prompt, gpt_model, temperature, client):
    completion = client.chat.completions.create(
        model=gpt_model,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    result = dict(completion.choices[0].message)['content']
    result = result.replace("\n", "").replace("{", "").replace("}", "").replace("\"", "").replace("[", "").replace("]",
                                                                                                                   "").strip()
    return [x.strip() for x in result.split(';')]


# Function to generate the prompt. There are three options:
# 1. ‘Simple’ prompt only provides the words and asks for a label which represents the topic key words.
# 2. 'Context' additionally provides some context including the nature of the text data and how the topics are constructed.
# 3. 'CoT' uses Chain-of-Thought prompting, including the provision of examples and review of labels
def generate_prompt(topics, type, nwords, ntopics):
    json_output = """{topics: { 
                            'topic': Topic0,
                            'label' : 'Label0',
                            'rationale': 'Rationale0'
                            }, 
                            {
                            topic': Topic1,
                            'label' : 'Label1',
                            'rationale': 'Rationale1'
                            }
            }"""
    if type == 'simple':
        prompt_template = """
            You are provided with the results of a topic modeling analysis. You are provided with the top {nwords} words for each of the {ntopics} topics. 
            Return only the labels and rationales for each topic. 
            Provide Your labels and rationales in JSON format like in the following example:
            {json_output}
            Provide your labels and rationales for all of the the following {ntopics} keyword lists in the order in which they appear: {topics}
            """
        prompt = prompt_template.format(
            **{"nwords": nwords, "ntopics": ntopics, "json_output": json_output, "topics": topics})
    elif type == 'context':
        prompt_template = """
                    For context, in a research project, a corpus of "Principal Risks and Uncertainties" sections from UK annual reports has been created. 
                    An LDA topic model has then been constructed based on the text. The analysis generated {ntopics} topics.
                    You are a research associate who is tasked with interpreting the output of topic models generated from corporate disclosures.
                    Your objective is to provide a label which best represents the semantic meaning of the topic. 
                    You are provided with the top {nwords} words for each of the {ntopics} topics. 
                    Return only the labels and rationales for each topic. 
                    Provide Your labels and rationales in JSON format like in the following example:
                    {json_output}
                    Provide your labels and rationales for all of the the following {ntopics} keyword lists: {topics}
                    """
        prompt = prompt_template.format(
            **{"nwords": nwords, "ntopics": ntopics, "json_output": json_output, "topics": topics})
    elif type == 'toc':
        prompt_template = """
                            For context, in a research project, a corpus of "Principal Risks and Uncertainties" sections from UK annual reports has been created. 
                            An LDA topic model has then been constructed based on the text. The analysis generated {ntopics} topics.
                            You are a research associate who is tasked with interpreting the output of topic models generated from corporate disclosures.
                            Your objective is to provide a label which best represents the semantic meaning of the topic.
                            Your goal is to review these keywords, generate specific labels for each topic, and ensure that the labels are mutually exclusive.
                            You are provided with the top {nwords} words for each of the {ntopics} topics.
                            Instructions:
                                1.  Read and Analyze Keywords:
                                1.a Carefully read the list of keywords for each of the 35 topics.
                                1.b Identify semantic links between the keywords within each topic.
                                2.  Generate Initial Labels:
                                2.a Based on your analysis, generate a specific and descriptive label for each topic.
                                2.b Labels should reflect the specific nature of the risks and uncertainties rather than generic terms like 'risk management' or 'risks.'
                                3.  Ensure Mutually Exclusive Labels:
                                3.a Compare topics with similar labels.
                                3.b Identify subtle differences between the topics.
                                3.c Modify the labels to ensure each one is unique and mutually exclusive.
                                Example: 
                                Input word lists
                                Topic 1: ['regulation', 'compliance', 'legal', 'policy', 'law', 'rules', 'regulatory', 'governance', 'legislation', 'audit']
                                Topic 2: ['market', 'competition', 'demand', 'consumer', 'price', 'sales', 'industry', 'growth', 'revenue', 'trend']

                                Topic 1:
                                •   Label: Regulatory Compliance Risks
                                •   Rationale: The keywords are centered around legal and regulatory aspects, indicating risks associated with compliance with laws and regulations. 
                                Topic 2:
                                •   Label: Market Competition Risks
                                •   Rationale: The keywords suggest risks related to market dynamics, competition, and consumer behavior affecting the company's performance.

                            Return only the labels and rationales for each topic. 
                            Provide Your labels and rationales in JSON format like in the following example:
                            {json_output}
                            Provide your labels and rationales for all of the the following {ntopics} keyword lists: {topics}
                            """
        prompt = prompt_template.format(
            **{"nwords": nwords, "ntopics": ntopics, "json_output": json_output, "topics": topics})
    else:
        prompt = "Error"
    return prompt


# Function to generate list of words for each topic
def generate_topic_word_list(df, nwords):
    words = ''
    for i in list(df.columns):
        words = words + i + ': ' + ' '.join(list(df.sort_values(i, ascending=False).index)[:nwords]) + '; '
    return words


# Function to execute the prompt
def get_labels(prompt, gpt_model, temperature, client):
    completion = client.chat.completions.create(
        model=gpt_model,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    result = dict(completion.choices[0].message)['content'].replace("'", '"')
    result = re.sub(r'(?<!")(\b\w+\b)(?!")\s*:', r'"\1":', result)
    result = re.sub(r':\s*(\b\w+\b)(?!")', r': "\1"', result)
    try:
        result = pd.json_normalize(json.loads(result), 'topics')
    except:
        result = pd.DataFrame([json.loads(result)])
    return result


# Include additional prompt which feeds in sentences
def generate_content_withSentences(ntopics, df, df2, nwords, topic_name=''):
    if ntopics == 1:
        topics = topic_name + ' (words): ' + ', '.join(list(df.sort_values(topic_name, ascending=False).index)[
                                                       :nwords]) + ' \n ' + topic_name + ' (sentences): ' + ' '.join(
            list(df2[topic_name]))
        content = """
                    Provide your label and rationale for the following keyword and sentence list: {topics}
                    """
        content = content.format(
            **{"nwords": nwords, "topics": topics})
    else:
        topics = ''
        for i in list(df.columns):
            topics = topics + i + ' (words): ' + ', '.join(list(df.sort_values(i, ascending=False).index)[
                                                           :nwords]) + ' \n ' + i + ' (sentences): ' + ' '.join(
                list(df2[i])) + '; '
        content = """
                        Include labels and rationales for all of the the following {ntopics} keyword lists: {topics}
                        """
        content = content.format(
            **{"nwords": nwords, "topics": topics, "ntopics": ntopics})
    return content.strip()


def generate_task_description_sentences(type, nwords, ntopics):
    if type == 'naive':
        task_description = """
                            You are provided with the results of a topic modeling analysis. 
                            You are provided with the index of the topic (e.g. Topic1), top {nwords} words for the topic and representative sentences. 
                            Please provide a label which best describes the topic and the rationale for the label.
                            Return only the label and rationale for the topic.
                            Do not use quotation marks in the label or rationale. 
                            Provide Your label and rationale as a string separated by a semicolon like in the following example:
                            [Label; Rationale]
                            """
        task_description = task_description.format(
            **{"nwords": nwords})
    else:
        json_output = """{topics: { 
                                   "topic": "Topic0",
                                   "label" : "Label0",
                                   "rationale": "Rationale0"
                                    }, 
                                    {
                                    "topic": "Topic1",
                                    "label" : "Label1",
                                    "rationale": "Rationale1"
                                    }
                            }"""
        if type == 'simple':
            task_description = """
                            You are provided with the results of a topic modeling analysis. 
                            You are provided with the top {nwords} words and representative sentences for each of the {ntopics} topics.
                            Return only the labels and rationales for each topic. 
                            Provide Your labels and rationales in JSON format like in the following example:
                            {json_output}
                            """
            task_description = task_description.format(
                **{"nwords": nwords, "ntopics": ntopics, "json_output": json_output})
        elif type == 'context':
            task_description = """
                        For context, in a research project, a corpus of 'Principal Risks and Uncertainties' sections from UK annual reports has been created. 
                        An LDA topic model has then been constructed based on the text. The analysis generated {ntopics} topics.
                        You are a research associate who is tasked with interpreting the output of topic models generated from corporate disclosures.
                        Your objective is to provide a label which best represents the semantic meaning of the topic. 
                        You are provided with the top {nwords} words and representative sentences for each of the {ntopics} topics. 
                        Return only the labels and rationales for each topic. 
                        Provide Your labels and rationales in JSON format like in the following example:
                        {json_output}                        
                        """
            task_description = task_description.format(
                **{"nwords": nwords, "ntopics": ntopics, "json_output": json_output})
        elif type == 'toc':
            task_description = """
                                For context, in a research project, a corpus of 'Principal Risks and Uncertainties' sections from UK annual reports has been created. 
                                An LDA topic model has then been constructed based on the text. The analysis generated {ntopics} topics.
                                You are a research associate who is tasked with interpreting the output of topic models generated from corporate disclosures.
                                Your objective is to provide a label which best represents the semantic meaning of the topic.
                                Your goal is to review these keywords, generate specific labels for each topic, and ensure that the labels are mutually exclusive.
                                You are provided with the top {nwords} words and representative sentences for each of the {ntopics} topics.
                                Instructions:
                                    1.  Read and Analyze Keywords:
                                    1.a Carefully read the list of keywords for each of the 35 topics.
                                    1.b Identify semantic links between the keywords within each topic.
                                    2.  Generate Initial Labels:
                                    2.a Based on your analysis, generate a specific and descriptive label for each topic.
                                    2.b Labels should reflect the specific nature of the risks and uncertainties rather than generic terms like 'risk management' or 'risks.'
                                    3.  Ensure Mutually Exclusive Labels:
                                    3.a Compare topics with similar labels.
                                    3.b Identify subtle differences between the topics.
                                    3.c Modify the labels to ensure each one is unique and mutually exclusive.
                                    Example: 
                                    Input word lists
                                    Topic 1: ['regulation', 'compliance', 'legal', 'policy', 'law', 'rules', 'regulatory', 'governance', 'legislation', 'audit']
                                    Topic 2: ['market', 'competition', 'demand', 'consumer', 'price', 'sales', 'industry', 'growth', 'revenue', 'trend']

                                    Topic 1:
                                    •   Label: Regulatory Compliance Risks
                                    •   Rationale: The keywords are centered around legal and regulatory aspects, indicating risks associated with compliance with laws and regulations. 
                                    Topic 2:
                                    •   Label: Market Competition Risks
                                    •   Rationale: The keywords suggest risks related to market dynamics, competition, and consumer behavior affecting the company's performance.

                                Return only the labels and rationales for each topic. 
                                Provide Your labels and rationales in JSON format like in the following example:
                                {json_output}
                                """
            task_description = task_description.format(
                **{"nwords": nwords, "ntopics": ntopics, "json_output": json_output})
        else:
            task_description = "Error"
    return task_description.strip()


# Function to check the labelling output to identify whether there are overlapping labels. If there are, the code
# asks GPT to review labels, rationales and keywords to generate revised topics
def check_and_fix_labels(label_df, words_weights, ntopics, client):
    check_labels = check_topic_labels(label_df, ntopics, client)

    if check_labels == 'NO OVERLAPPING TOPIC LABELS':
        print('Labels checked - no overlapping labels!')
        return label_df
    elif check_labels == 'TOO MANY TOPICS TO CHECK OVERLAP - MANUAL CHECK REQUIRED!':
        print(check_labels)
        return label_df
    else:
        print('Labels checked - overlapping labels identified!')
        print('Suggesting new labels...')
        problem_topics = pd.DataFrame(check_labels, columns=['topic', 'label'])
        problem_topics = pd.merge(problem_topics, label_df[['topic', 'rationale']], on='topic', how='left')
        keywords = []
        for t in list(problem_topics.topic):
            keywords = keywords + [', '.join(list(words_weights[t].sort_values(ascending=False).index)[:10])]
        problem_topics['keywords'] = keywords
        revised_labels = pd.DataFrame(fix_topic_labels(problem_topics, client, ntopics), columns=['topic', 'label', 'rationale'])
        for t in list(problem_topics.topic):
            label_df.loc[label_df.topic == t, 'label'] = \
                list(revised_labels.loc[revised_labels.topic == t, 'label'])[0]
            label_df.loc[label_df.topic == t, 'rationale'] = \
                list(revised_labels.loc[revised_labels.topic == t, 'rationale'])[0]
        print('Done!')
        return label_df


# Function to identify non-mutually exclusive topic labels
def check_topic_labels(label_df, ntopics, client):
    label_df['combined'] = label_df['topic'] + ': ' + label_df['label']
    labels_as_string = 'Consider each of the following topic labels: ' + '; '.join(label_df['combined'])
    task_description = """
                            For context, in a research project, a corpus of 'Principal Risks and Uncertainties' sections from UK annual reports has been created. 
                            An LDA topic model has then been constructed based on the text. The analysis generated {ntopics} topics.
                            You are a research associate who is tasked with interpreting the output of topic models generated from corporate disclosures.
                            You have already provided a label which best represents the semantic meaning of the topic.
                            Your objective now is to review the suggested labels and identify whether the labels are mutually exclusive.
                            Topic labels should not be considered mutually exclusive if they share close synonyms or root words (e.g. 'financial risk' 'finance risk' and 'financing risk' are not mutually exclusive)
                            Instructions:
                                1.  Read and Analyze topic labels for each of the {ntopics}.:
                                2.  Identify overlap in suggested labels.
                                3.  Return all topic indices and labels for each group of overlapping topics.
                                3.i. If there are two topics which share similar labels, return the index and label for both topics
                                3.ii. If there are three topics which share similar labels, return the index and label for all three topics
                                4. Check that the number of topics you are returning is equal to the total number of topics which has an overlapping label
                                5.  If there are no overlapping labels, return only 'NO OVERLAPPING TOPIC LABELS'

                            Example: 
                            Input topic labels:
                            Topic 1: 'Oil and gas risk'
                            Topic 2: 'Interest rate risk'
                            Topic 3: 'Gas and oil risk'
                            Topic 4: 'Pension risk'
                            Topic 5: 'Pension risk'
                            Topic 6: 'Gas and oil drilling risk'

                            Answer:
                            Topic0, Oil and gas risk
                            Topic3, Gas and oil risk
                            Topic6, "Gas or oil risk
                            Topic4, "Pension risk
                            Topic5, "Pension risk

                            Explanation:
                            Topic0, Topic3 and Topic6 are very similar in that the only difference is word order or the replacement of 'and' with 'or'.
                            Topic4 and Topic5 are identical.
                            Therefore, you should return all indices (Topic0, Topic3, Topic6, Topic4 and Topic5) as well as the labels. 

                            Final rules:
                            Return only the topic index and topic labels for each overlapping topic according to the JSON format below.
                            Do not provide any other information, such as rationales or groupings of overlapping topics. 
                            Provide Your labels exactly in JSON format as follows: {json_output}

                            """
    json_output = """{overlapping_topics: { 
                                   "topic": "Topic0",
                                   "label" : "Label0"
                                    }, 
                                    {
                                    "topic": "Topic2",
                                    "label" : "Label2"
                                    }
                            }"""
    task_description = task_description.format(**{"ntopics": ntopics, "json_output": json_output})
    message_history = []
    message_history.append({"role": "system", "content": task_description})
    message_history.append({"role": "user", "content": labels_as_string})
    try:
        response0 = client.chat.completions.create(
            model="gpt-4",
            #response_format={"type": "json_object"},
            max_tokens=4096,
            messages=message_history)
        if dict(response0.choices[0].message)['content'] == 'NO OVERLAPPING TOPIC LABELS':
            output = 'NO OVERLAPPING TOPIC LABELS'
        else:
            #print(dict(response0.choices[0].message)['content'])
            output = json.loads(dict(response0.choices[0].message)['content'])['overlapping_topics']
    except openai.OpenAIError:
        output = 'TOO MANY TOPICS TO CHECK OVERLAP - MANUAL CHECK REQUIRED!'
    return output


# Function to generate mutually exclusive topic labels
def fix_topic_labels(label_df, client, ntopics):
    label_df['combined'] = label_df['topic'] + ': label - ' + label_df['label'] + ', rationale - ' + label_df[
        'rationale'] + ', keywords - (' + label_df['keywords'] + ')'
    labels_as_string = 'Consider each of the following problematic topic labels: ' + '; '.join(label_df['combined'])
    task_description = """
                            For context, in a research project, a corpus of 'Principal Risks and Uncertainties' sections from UK annual reports has been created. 
                            An LDA topic model has then been constructed based on the text. The analysis generated {ntopics} topics.
                            You are a research associate who is tasked with interpreting the output of topic models generated from corporate disclosures.
                            You have already provided a label which best represents the semantic meaning of the topic.
                            You have already reviewed the suggested labels and identify whether the labels are mutually exclusive.
                            You will now be presented with topics for which you have previously identified as being too similar.
                            It is your task to read the topic labels, the rationale you provided for the label, and the top ten words associated with the topic.
                            Your goal is to review these keywords, generate specific labels for each topic, and ensure that the labels are mutually exclusive.
                            You will then provide the revised set of topic words.
                            Instructions:
                                1.  Read and Analyze topic labels for each of the {ntopics}.
                                2.  Identify overlap in suggested labels.
                                3.  Carefully read the rationales and keywords associated with each topic.
                                4.  Ensure Mutually Exclusive Labels:
                                4.a Compare topics with similar labels.
                                4.b Identify subtle differences between the topics.
                                4.c Modify the labels to ensure each one is unique and mutually exclusive.Return all topic indices and labels for each group of overlapping topics.

                            Data will be presented to you in the following format:
                            "TopicN: label - ORIGINAL LABEL, rationale - RATIONALE FOR ORIGINAL LABEL, keywords - TOP TEN KEY WORDS"

                            Final rules:
                            Return only the topic index, revised topic labels and rationale for the revised label.
                            Do not provide any other information, such as groupings of overlapping topics. 
                            Provide Your labels in JSON format exactly as follows: {json_output}

                            """
    json_output = """{revised_labels: { 
                                   "topic": "Topic0",
                                   "label" : "Label0",
                                   "rationale" : "Rationale0"
                                    }, 
                                    {
                                    "topic": "Topic1",
                                    "label" : "Label1",
                                    "rationale" : "Rationale1"
                                    }
                            }"""
    task_description = task_description.format(
        **{"ntopics": ntopics, "json_output": json_output})
    message_history = []
    message_history.append({"role": "system", "content": task_description})
    message_history.append({"role": "user", "content": labels_as_string})
    response0 = client.chat.completions.create(
        model="gpt-4",
        #response_format={"type": "json_object"},
        max_tokens=4096,
        messages=message_history)
    output = json.loads(dict(response0.choices[0].message)['content'])['revised_labels']
    return output





def labelling(savepath, client, gpt_model="gpt-4o"):
    # retreive phi (topic-term matrix)
    model, textbeforetokenization, tokenized_text, corpus, id2word = load_lda_run(savepath)

    phi = model.get_topics()
    phi = pd.DataFrame(phi)
    phi = phi.rename(columns=id2word.id2token)
    phi = phi.T
    
    # number of topics
    ntopics = len(phi.columns)

    # Create dataframe where the index contains words, columns are topics and values are weights
    phi.columns = [f'Topic{x}' for x in list(phi.columns)]

    # Moving to the ToC approach (including splitting into chunks)
    task_description = generate_task_description('toc', 10, ntopics)

    # Splitting into chunks to ensure all topics are included
    chunk_size = 5
    chunks = chunk_topics(phi, chunk_size)

    # List to store intruder word identified by GPT
    topic_labels = []

    # Apply GPT
    for c in range(chunks):
        cols = [f'Topic{z}' for z in range(c * chunk_size, c * chunk_size + chunk_size, 1)]
        cols = [z for z in cols if z in list(phi.columns)]
        chunk = phi[cols]

        content = generate_content(chunk_size, chunk, 10, '')
        message_history = []
        message_history.append({"role": "system", "content": task_description})
        message_history.append({"role": "user", "content": content})
        response0 = client.chat.completions.create(
            model=gpt_model,
            response_format={"type": "json_object"},
            max_tokens=4096,
            messages=message_history)
        labels_toc = pd.json_normalize(json.loads(dict(response0.choices[0].message)['content']), 'topics')
        topic_labels = topic_labels + [labels_toc]
    labels_toc = pd.concat(topic_labels).reset_index(drop=True)

    # Checking for overlapping labels
    labels_toc_revised = check_and_fix_labels(labels_toc, phi, ntopics, client)

    # Generate the list of top 10 words by weight for each topic
    top_words_per_topic = []
    for topic in phi.columns:
        # Get the top 10 words for this topic
        top_words = ', '.join(phi[topic].nlargest(10).index.tolist())
        top_words_per_topic.append(top_words)
    labels_toc_revised['keywords'] = top_words_per_topic

    return labels_toc_revised[['topic', 'label', 'rationale', 'keywords']]