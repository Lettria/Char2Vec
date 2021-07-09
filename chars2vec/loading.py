import os
import sys
import json
import nmslib
import pickle
import numpy as np

from .model import Chars2Vec

def load_model(path = ''):
    '''
    Loading pretrained model
    Input:  path, folder containing model (must contain weights.h5 and model.pkl)
    Return: c2v_model trained model
    '''
    path_to_model = os.path.dirname(os.path.abspath(__file__)) + '/trained_model/' + path

    with open(path_to_model + '/model.pkl', 'rb') as f:
        structure = pickle.load(f)
        emb_dim, char_to_ix = structure[0], structure[1]

    c2v_model = Chars2Vec(emb_dim, char_to_ix)
    try:
        c2v_model.model.load_weights(path_to_model + '/weights.h5')
        print("Model and weights loaded successfully " + path)
    except Exception as e:
        print(e)
        raise e
    c2v_model.embedding_model.compile(optimizer='adam', loss='mae')
    return c2v_model

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def vectorize_dict(path, emb_size = 150, encoding_type = 'utf-8'):
    """ Vectorization of dictionnary
        Input:  Path of dictionnary to embed (Either json list or txt with 1 word per line)
        Output: dict_vectorized{}.txt        dict_index_{}.bin"""

    c2v_model = load_model("train_fr_{}".format(emb_size))

    with open(path, "r", encoding = encoding_type) as file:
        if path.endswith(".json"):
            words = json.load(file)
        else:
            words = []
            for line in file:
                words.append(line.strip())

    for i in range(0, len(words), 50000):
        batch = np.array(c2v_model.vectorize(words[i:i + 50000]))
        if i == 0:
            embs = batch
        else:
            embs = np.concatenate([embs, batch])
        print("Vectorizing: " + str(embs.shape) + "...")
    with open("./dict_vectorized_{}.txt".format(emb_size), "w") as file:
        json.dump(embs, file, cls=NumpyEncoder)

def create_index(emb_size = 150):
    """ Creates nmslib index allowing knn search """
    with open("./dict_vectorized_{}.txt".format(emb_size), "r") as file:
        embs = json.load(file)

    index = nmslib.init(method="hnsw", space="cosinesimil")
    index.addDataPointBatch(embs)
    index.createIndex({'post':2}, print_progress = False)
    index.saveIndex('./dict_index_{}.bin'.format(emb_size), save_data=True)
