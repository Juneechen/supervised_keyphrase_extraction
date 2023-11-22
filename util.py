import nltk
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
    '''remove [, ], ', from the string, and split into list of tokens by comma. 
    Apply nltk.word_tokenize to each token for further cleaning'''
    phrases = keywords.replace('[', '').replace(']', '').replace('\'', '').split(', ')
    return [nltk.word_tokenize(phrase) for phrase in phrases]


def keywords_marking_exact(keyword_phrases: list, sequence: list, max_len: int, tokenizer: Tokenizer):
    '''
    Mark a sequence of tokens for the exact keyword phrases. 
    If the keyword phrase as a whole is not in the sequence, it will not be marked.
    
    params:
        keyword_phrases: a list of keyword phrases, each phrase is a list of tokens
    return:
        a list of binary labels of the same length as input sequence, 1 for keyword phrases, 0 for others
    '''
    print("phrases", keyword_phrases)
    print(">>>")
    binary_labels = [0] * max_len

    # for phrase in keyword_phrases:
    #     if phrase == []:
    #         continue
    #     print("phrase", phrase)
    #     # convert phrase tokens to sequence tokens
    #     phrase_tokens = tokenizer.texts_to_sequences(phrase) # [0] because texts_to_sequences returns a list of lists
    #     print("phrase tokens", phrase_tokens)
    #     print()
    #     # see if the phrase is in the sequence
    #     for i in range(len(sequence) - len(phrase_tokens) + 1):
    #         if sequence[i:i+len(phrase_tokens)] == phrase_tokens: # matching the whole phrase
    #             binary_labels[i:i+len(phrase_tokens)] = [1] * len(phrase_tokens)

    phrase_tokens = tokenizer.texts_to_sequences(keyword_phrases)
    print("phrase tokens", phrase_tokens)

    return binary_labels

def keywords_marking(keywords: list, sequences: list, max_len: int, tokenizer: Tokenizer):
    binary_labels = []
    for i in range(len(keywords)):
        binary_labels.append(keywords_marking_exact(keywords[i], sequences[i], max_len, tokenizer))
    return binary_labels