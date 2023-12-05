import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
# import tf.keras.preprocessing.sequence.pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# read data from csv into dataframe
def read_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # handle 'utf-8' codec can't decode
        df = pd.read_csv(file_path, encoding='latin-1')

    return df

def clean_text(text):
    '''
    clean text by removing non-alphabetic characters and stop words
    '''
    text = text.lower().strip()

    # keeping non-alphabetic char, space, and hyphen
    text = re.sub('[^\w\s-]', '', text) 

    # tokenize
    tokens = word_tokenize(text)
          
    # remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # # join tokens back into string
    # text = ' '.join(tokens)

    return tokens

def clean_kp(kp_str: str):
    # remove special char
    kp_str = kp_str.lower().strip()
    kp_str = re.sub(r'[^\w\s,-]', '', kp_str)

    # split into a list of keyphrases
    keyphrases = kp_str.split(',')

    # lemmatize tokens in keyphrases
    lemmatizer = WordNetLemmatizer()
    lemmatized_kps = []
    for kp in keyphrases:
        kp_tokens = kp.split()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in kp_tokens]
        lemmatized_kps.append(' '.join(lemmatized_tokens))

    return lemmatized_kps

def setup_tokenizer(train_df, test_df, columns):
    '''
    set up a keras tokenizer with text from the given columns
    '''
    tokenizer = Tokenizer()
    for col in columns:
        tokenizer.fit_on_texts(train_df[col])
        tokenizer.fit_on_texts(test_df[col])
    
    # set tokenizer index 0 to be the padding token
    tokenizer.index_word[0] = '<PAD>'
    tokenizer.word_index['<PAD>'] = 0

    return tokenizer

def get_embeddings_matrix(tokenizer, embeddings, emb_dim):
    '''
    get the embeddings matrix for the given tokenizer and embeddings
    '''
    word_index = tokenizer.word_index
    vocab_size = len(word_index) 

    embedding_matrix = np.zeros((vocab_size, emb_dim)) # create a matrix with all zeros
    for word, i in word_index.items():
        if word in embeddings:
            embedding_vector = embeddings[word]
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def tokens_to_embeddings(tokens, tokenizer, embeddings_matrix, emb_dim, max_len):
    '''
    convert tokens representing one sample to embeddings; padding and truncation are applied
    return:
        a list of embeddings of len=max_len;
        each embedding is a list of len=emb_dim
    '''

    embeddings_list = []
    
    # convert the tokens to integers sequences (with padding)
    tokens_idx = tokenizer.texts_to_sequences([tokens])
    tokens_idx = pad_sequences(tokens_idx, maxlen=max_len, padding='post', truncating='post')

    for i in tokens_idx[0]:
        embeddings_list.append(embeddings_matrix[i])

    # print("embedding list shape:", len(embeddings_list))
    # print("-----------next sample:-----------")

    return embeddings_list

def mark_keywords(keyphrases: list, input_tokens: list, max_len: int):
    '''
    mark a sequence of tokens for the partial keyword phrases. 
    If any part of a keyword phrase is in the sequence, it will be marked as 1.
    
    params:
        keyphrases: a list of preprocessed keyphrases, in the format of "phrase 1, phrase 2, ..."
        input_tokens: a list of tokens represeting one sample input
        max_len: maximum length of the input
    return:
        a list of binary labels, length = sample length = max_len;
        1 for keyword, 0 for non-keyword
    '''
    label = [0] * max_len

    kp_tokens = []
    for kp in keyphrases:
        kp_tokens.extend(kp.split())

    for i in range(len(input_tokens)):
        if i >= max_len:
            break
        if input_tokens[i] in kp_tokens:
            label[i] = 1
            
    return label

def make_labels(keyphrases_df_col, input_tokens_df_col, max_len):
    '''
    create label with the given keyphrases and input tokens.

    params:
        keyphrases_df_col: a dataframe column of keyphrases, preprocessed list of phrases
        input_tokens_df_col: a dataframe column of input tokens
        tokenizer: a keras tokenizer
        max_len: maximum length of each input sample
    return:
        a numpy array of labels, shape: (number of samples, sample length = max_len, 1)
    '''
    labels = []

    for keyphrases, input_tokens in zip(keyphrases_df_col, input_tokens_df_col):
        labels.append(mark_keywords(keyphrases, input_tokens, max_len))
    
    return np.asarray(labels)

def preprocess_data(df, input_cols: list, label_col: str, sample_size=None):
    # sample the dataframe for testing on part of the data
    if sample_size is not None:
        df = df.sample(n=sample_size)

    # concatenate the columns e.g. ['title', 'abstract']
    df['input_tokens'] = df[input_cols].agg(' '.join, axis=1)

    # preprocess and tokenize the combined column
    df['input_tokens'] = df['input_tokens'].apply(lambda row: clean_text(row))

    # clean keyphrases
    df['clean_kp'] = df[label_col].apply(lambda row: clean_kp(row))

    return df

def create_input_array(df, input_col: str, kp_col: str, tokenizer, embeddings, emb_dim, max_len):
    """
    Convert samples from dataframe to input and label numpy arrays.
    Each input is a list of embeddings for each token in the sample; unknown words represented by a vector of 0s.   
    Input array shape: (number of samples, sample length = max_len, embedding dimension)
    Label array shape: (number of samples, sample length = max_len, 1)

    params:
        df: dataframe
        input_col: column to be used as input
        embeddings: glove embeddings
        max_len: maximum length of the input
    return: 
        numpy array of inputs, numpy array of labels
    """
    # create the input array
    input_array = []

    for sample_tokens in df[input_col]:
        # convert to a list of embeddings (with padding and truncation)
        embeddings_list = tokens_to_embeddings(sample_tokens, tokenizer, embeddings, emb_dim, max_len)
        input_array.append(embeddings_list)

    # create labels
    labels = make_labels(df[kp_col], df[input_col], max_len)

    # return the input array as a numpy array
    return np.array(input_array), labels

# # convert prediction to keywords
# def pred_to_keywords(pred, input_seq, tokenizer):
#     # convert prediction to binary with a threshold of 0.5
#     threshold = 0.5
#     binary_pred = (pred > threshold).astype(int)
    
#     # print()
#     # print(binary_pred)
#     # convert binary prediction to keywords
#     keywords = []
#     for i in range(len(binary_pred)):
#         # break reaches the end of the sequence before the padding part
#         if input_seq[i] == 0:
#             break
#         if binary_pred[i] == 1:
#             # print("keyword at:", i)
#             # print("int rep.:", input_seq[i])
#             keywords.append(tokenizer.index_word[input_seq[i]])
    
#     return keywords

# convert prediction to keywords
def pred_to_keywords(preds, input_tokens):
    '''
    Retrieve words from input_tokens that are predicted to be keywords.

    params:
        preds: a list of list of predictions for each word in the input sequence
        input_tokens: a list of list of tokens representing the original input sequence (without padding)
    return:
        a list of list of keywords
    '''
    keywords = []
    # print("In Utils - pred")
    for i in range(len(preds)):
        pred = preds[i]
        input_seq = input_tokens[i]

        # convert prediction to binary with a threshold of 0.5
        threshold = 0.42
        binary_pred = (pred > threshold).astype(int)
        # print("binary_pred:", binary_pred)

        kws = []
        for j in range(len(binary_pred)):   # fix length = max_len
            # break reaches the end of the sequence (without padding)
            if j >= len(input_seq):
                break
            if binary_pred[j] == 1:
                # print("keyword at:", j)
                kws.append(input_seq[j])

        keywords.append(kws)
    
    return keywords