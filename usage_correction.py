import os
import json
import nmslib
import sys

from chardet import detect
import chars2vec as c2v

dict_path = "example_dict.txt"

with open(dict_path, 'rb') as file:
    raw = file.read()
    encoding_type = detect(raw)['encoding']

""" Vectorize dictionnary, creates and index through nmslib. Only done the first time """
emb_size = 150
if not os.path.exists("dict_vectorized_{}.txt".format(emb_size)):
    c2v.vectorize_dict(dict_path, encoding_type = encoding_type)
if not os.path.exists('./dict_index_{}.bin'.format(emb_size)):
    c2v.create_index(emb_size)

"""Loading necessary resources"""
dictionnary = []
with open(dict_path, 'r', encoding = encoding_type) as file:
    for line in file:
        dictionnary.append(line.strip())
        if dict_path.endswith(".json"):
            dictionnary = json.load(file)
        else:
            dictionnary = []
            for line in file:
                dictionnary.append(line.strip())

index = nmslib.init(method="hnsw", space="cosinesimil")
index.loadIndex('./dict_index_{}.bin'.format(emb_size))
c2v_model = c2v.load_model("train_fr_150")

import time

if len(sys.argv) == 1:
    """ K-nearest-neigbors search"""
    print("\nEdit distance 1:")
    stamp = time.time()
    requests1 = []
    requests1.append(c2v.find_knn("langage", dictionnary, c2v_model, index))
    requests1.append(c2v.find_knn("langqge", dictionnary, c2v_model, index))
    requests1.append(c2v.find_knn("langagee", dictionnary, c2v_model, index))
    time1 = (time.time() - stamp)

    print("langage", requests1[0])
    print("langqge", requests1[1])
    print("langagee", requests1[2])
    print("\nMean time by request: " + str(time1/ 3.0))

    print("\nEdit distance 2:")
    stamp = time.time()
    requests2 = []
    requests2.append(c2v.find_knn("langage", dictionnary, c2v_model, index, distance = 2))
    requests2.append(c2v.find_knn("langqge", dictionnary, c2v_model, index, distance = 2))
    requests2.append(c2v.find_knn("langagee", dictionnary, c2v_model, index, distance = 2))
    time2 = (time.time() - stamp)

    print("langage", requests2[0])
    print("langqge", requests2[1])
    print("langagee", requests2[2])
    print("\nMean time by request: " + str(time2/ 3.0))
else:
    print("\nEdit distance 1:")
    for i in range(1, len(sys.argv)):
        print(sys.argv[i], c2v.find_knn(sys.argv[i], dictionnary, c2v_model, index))
    print("\nEdit distance 2:")
    for i in range(1, len(sys.argv)):
        print(sys.argv[i], c2v.find_knn(sys.argv[i], dictionnary, c2v_model, index, distance = 2))
