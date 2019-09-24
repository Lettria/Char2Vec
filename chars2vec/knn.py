from .levenshtein import levenshtein

""" Allows filtering of words according to maximum edit distance"""
def choose_size(word):
    if len(word) <= 6:
        return 1
    else:
        return 2

""" Returns k nearest neighbors according to spelling.
    Input:
    word:           Word to search
    dictionnary:    Dictionnary used for research (must be the same than used for building the index)
    c2v_model:      char2vec model for vectorizing string
    index:          Search index built with nmslib on vectorized dict"""
def find_knn(word, dictionnary, c2v_model, index, k = 80, filter = 1, distance = 1):
    ids, distances = index.knnQuery(c2v_model.vectorize([word]), k)
    if filter:
        knn = [dictionnary[i] for i in ids if levenshtein(word, dictionnary[i]) <= distance]
    else:
        knn = [dictionnary[i] for i in ids]
    if len(knn) == 0:
        knn = [dictionnary[i] for i in ids if levenshtein(word, dictionnary[i]) <= distance + 1]
    return knn
