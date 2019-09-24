import sys
import numpy as np
import json
import nmslib
sys.path.append('..')
from chars2vec.model import load_model

""" Dictionnary vectorization and index creation for efficient knn search
    Output:
    dict_vectorized{}.txt
    dict_index_{}.bin"""

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

emb_size = 150
c2v_model = load_model("train_fr_{}".format(emb_size))

with open("../corpus/all_words_traite.txt", "r") as file:
    words = json.load(file)

# Create word embeddings
for i in range(0, 600000, 50000):
    batch = np.array(c2v_model.vectorize(words[i:i + 50000]))
    print(batch)
    if i == 0:
        embs = batch
    else:
        embs = np.concatenate([embs, batch])
    print(embs.shape)
with open("./index_data/dict_vectorized_{}.txt".format(emb_size), "w") as file:
    json.dump(embs, file, cls=NumpyEncoder)

index = nmslib.init(method="hnsw", space="cosinesimil")

index.addDataPointBatch(embs)
index.createIndex({'post':2}, print_progress = False)
index.saveIndex('./index_data/dict_index_{}.bin'.format(emb_size), save_data=True)
