from chars2vec import Chars2Vec, DataGenerator
from numpy.random import seed
from keras.callbacks import Callback
import json
import tensorflow as tf
seed(42)

""" Mute tensorflow warning """
tf.logging.set_verbosity(tf.logging.ERROR)

""" Training script for the model, the model takes as input a list of accepted characters (model_chars) and a data generator.
    The script provides the possibility to stop and resume training."""

with open("./chars2vec/training_data/training_data.json", "r") as file:
    r = json.load(file)
print("File loaded")

emb_dim = 150
path_to_model = './trained_model/train_fr_{}'.format(emb_dim)

model_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
               '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<',
               '=', '>', '?', '@', '_', 'a', 'à', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
               'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
               'x', 'y', 'z']

char_to_ix = {ch: i for i, ch in enumerate(model_chars)}
c2v_model = Chars2Vec(emb_dim, char_to_ix)
c2v_model.load_weights("train_fr_{}".format(emb_dim))

#with generator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
callbacks = []
filepath= path_to_model + "/weights-improvement-{epoch:02d}-{loss:.5f}.hdf5"

"""Model saving at epoch end"""
class Checkpoint(Callback):
    def on_epoch_end(self, batch, logs = {}):
        c2v_model.save_model("train_fr_{}".format(emb_dim))

callbacks.append(Checkpoint())
callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.7, patience=3, verbose=0, min_lr=0.0002))

X_train, y_train = r[0], r[1]
data_gen = DataGenerator(X_train, y_train, char_to_ix, batch_size = 128, emb_dim = emb_dim)
c2v_model.model.fit_generator(data_gen, epochs = 50, callbacks = callbacks)

c2v_model.save_model(path_to_model)
