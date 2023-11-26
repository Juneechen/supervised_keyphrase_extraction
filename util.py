import nltk
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

# create binary label for keywords in the sentence
def create_label(sentence, keywords, max_len):
    label = [0] * max_len
    for keyword in keywords:
        if keyword in sentence:
            start = sentence.index(keyword)
            end = start + len(keyword)
            label[start:end] = [1] * (end - start)
    return label

# read data from csv into dataframe
def read_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # handle 'utf-8' codec can't decode
        df = pd.read_csv(file_path, encoding='latin-1')

    return df

# def read_into_list(file_path):
#     data = []
#     try:
#         df = pd.read_csv(file_path, encoding='utf-8')
#     except UnicodeDecodeError:
#         # handle 'utf-8' codec can't decode
#         df = pd.read_csv(file_path, encoding='latin-1')

#     return df.values.tolist()

# tokenize sentence from a column in df into words
def tokenize_sentence(df_column):
    tokenized_sentences = []
    for sentence in df_column:
        tokenized_sentences.append(nltk.word_tokenize(sentence))
    return tokenized_sentences

def clean_keywords(keywords: str):
    '''remove [, ], ', from the string'''
    # phrases = keywords.replace('[', '').replace(']', '').replace('\'', '').split(', ')
    phrases = keywords.replace('[', '').replace(']', '').replace('\'', '').strip()
    # return [nltk.word_tokenize(phrase) for phrase in phrases]
    return phrases


def mark_exact(keyword_phrases: list, sequence: list, max_len: int, tokenizer: Tokenizer):
    '''
    Mark a sequence of tokens for the exact keyword phrases. 
    If the keyword phrase as a whole is not in the sequence, it will not be marked.
    
    params:
        keyword_phrases: a list of keyword phrases, each is a string formatted as "phrase1, phrase2, ..."
    return:
        a list of binary labels of the same length as input sequence, 1 for keyword phrases, 0 for others
    '''
    # print("phrases", keyword_phrases)
    # print(">>>")
    binary_labels = [0] * max_len

    # convert keyword phrases to tokens
    # print("keyword_phrase:", keyword_phrases)
    phrases = keyword_phrases.split(', ')
    # print("phrases:", phrases)
    phrase_tokens = tokenizer.texts_to_sequences(phrases)
    # print("phrase tokens", phrase_tokens)

    # mark the sequence
    for i in range(len(sequence) - len(phrase_tokens) + 1):
        if sequence[i:i+len(phrase_tokens)] == phrase_tokens: # matching the whole phrase
            binary_labels[i:i+len(phrase_tokens)] = [1] * len(phrase_tokens)

    # print(binary_labels)
    return binary_labels

def mark_partial(keyword_phrases: list, sequence: np.ndarray, max_len: int, tokenizer: Tokenizer):
    '''
    mark a sequence of tokens for the partial keyword phrases. 
    If any part of the keyword phrase is in the sequence, it will be marked.
    
    params:
        keyword_phrases: a list of keyword phrases, each is a string formatted as "phrase1, phrase2, ..."
    return:
        a list of binary labels of the same length as input sequence, 1 for keyword phrases, 0 for others
    '''
    binary_labels = [0] * max_len

    # convert keyword phrases to tokens
    # print("keyword_phrase:", keyword_phrases)
    phrases = keyword_phrases.split(', ')
    # print("phrases:", phrases)
    phrase_tokens = tokenizer.texts_to_sequences(phrases)
    # print("phrase_tokens:", phrase_tokens)

    sequence = sequence.tolist()
    # mark the sequence
    for phrase in phrase_tokens:
        for token in phrase:
            if token in sequence:
                # print("token:", token)
                # print("at:", sequence.index(token))
                try:
                    binary_labels[sequence.index(token)] = 1
                except IndexError:
                    print(f"IndexError: token {token} at {sequence.index(token)}; \
                          sequence length: {len(sequence)}")

    # print(binary_labels)    
    return binary_labels

def keywords_marking(keywords: list, sequences: list, max_len: int, tokenizer: Tokenizer):
    binary_labels = []
    for i in range(len(keywords)):
        binary_labels.append(mark_partial(keywords[i], sequences[i], max_len, tokenizer))
    return np.asarray(binary_labels)

# convert prediction to keywords
def pred_to_keywords(pred, input_seq, tokenizer):
    # convert prediction to binary with a threshold of 0.5
    threshold = 0.5
    binary_pred = (pred > threshold).astype(int)
    
    # print()
    # print(binary_pred)
    # convert binary prediction to keywords
    keywords = []
    for i in range(len(binary_pred)):
        # break reaches the end of the sequence before the padding part
        if input_seq[i] == 0:
            break
        if binary_pred[i] == 1:
            # print("keyword at:", i)
            # print("int rep.:", input_seq[i])
            keywords.append(tokenizer.index_word[input_seq[i]])
    
    return keywords