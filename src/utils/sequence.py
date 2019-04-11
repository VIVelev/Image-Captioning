from string import punctuation

import numpy as np

from .config import PATH_TO_DATA

__all__ = [
    'init_word2idx',
    'init_idx2word',
    'clean',
    'load_word_embedding_map',
    'init_word_embedding_matrix',
]


def init_word2idx(vocabulary):
    return {val: key for key, val in enumerate(vocabulary)}
    
def init_idx2word(vocabulary):
    return {key: val for key, val in enumerate(vocabulary)}

def clean(sentence):
    # Tokenize
    tokens = sentence.split()
    
    # Lower Case
    tokens = [token.lower() for token in tokens]
    
    # Remove punct
    for i in range(len(tokens)):
        tokens[i] = ''.join([ch for ch in tokens[i] if ch not in punctuation])
    
    # Remove hanging chars
    tokens = [token for token in tokens if len(token) > 1 or token == 'a']
    
    # Remove tokens with digits in it
    tokens = [token for token in tokens if token.isalpha()]
    
    return ' '.join(tokens)

def load_word_embedding_map(path_to_embeddings=PATH_TO_DATA+'glove.6B/glove.6B.300d.txt'):
    word_embedding = dict()

    with open(path_to_embeddings, encoding='utf-8') as f_embeddings:
        for line in f_embeddings:
            values = line.split()
            word_embedding[values[0]] = np.asarray(values[1:], dtype='float64')
    
    return word_embedding

def init_word_embedding_matrix(dim, voc_size, word2idx, path_to_embeddings=PATH_TO_DATA+'glove.6B/glove.6B.300d.txt'):
    word_embedding = load_word_embedding_map(path_to_embeddings=path_to_embeddings)
    embedding_matrix = np.zeros((voc_size, dim))

    for word, idx in word2idx.items():
        try:
            embedding_matrix[idx, :] = word_embedding[word]
        except KeyError:
            pass
        
    return embedding_matrix
