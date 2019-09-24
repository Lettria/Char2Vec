import numpy as np
import random
import pickle
import keras
import os
np.random.seed(42)

class Chars2Vec:

    def __init__(self, emb_dim, char_to_ix):
        '''
        :param emb_dim: int, dimension of embeddings.
        :param char_to_ix: dictionnary of authorized characters,
            keys = characters, values = sequence numbers of characters.
        '''

        if not isinstance(emb_dim, int) or emb_dim < 1:
            raise TypeError("parameter 'emb_dim' must be a positive integer")

        if not isinstance(char_to_ix, dict):
            raise TypeError("parameter 'char_to_ix' must be a dictionary")

        self.char_to_ix = char_to_ix
        self.ix_to_char = {char_to_ix[ch]: ch for ch in char_to_ix}
        self.vocab_size = len(self.char_to_ix)
        self.dim = emb_dim
        self.cache = {}

        lstm_input = keras.layers.Input(shape=(None, self.vocab_size))

        x = keras.layers.LSTM(emb_dim, return_sequences=True)(lstm_input)
        x = keras.layers.LSTM(emb_dim)(x)

        self.embedding_model = keras.models.Model(inputs=[lstm_input], outputs=x)

        model_input_1 = keras.layers.Input(shape=(None, self.vocab_size))
        model_input_2 = keras.layers.Input(shape=(None, self.vocab_size))

        embedding_1 = self.embedding_model(model_input_1)
        embedding_2 = self.embedding_model(model_input_2)
        x = keras.layers.Subtract()([embedding_1, embedding_2])
        x = keras.layers.Dot(1)([x, x])
        model_output = keras.layers.Dense(1, activation='sigmoid')(x)

        self.model = keras.models.Model(inputs=[model_input_1, model_input_2], outputs=model_output)
        self.model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mae')


    def fit(self, word_pairs, targets,
            max_epochs, patience, validation_split, batch_size):
        '''
        Fits model.
        :param word_pairs: list or numpy.ndarray of word pairs.
        :param targets: list or numpy.ndarray of targets.
        :param patience: parameter 'patience' of callback in keras model.
        '''
        x_1, x_2 = [], []
        for pair_words in word_pairs:
            emb_list_1, emb_list_2 = [], []
            if not isinstance(pair_words[0], str) or not isinstance(pair_words[1], str):
                raise TypeError("word must be a string")
            first_word = pair_words[0].lower()
            second_word = pair_words[1].lower()
            for t in range(len(first_word)):
                if first_word[t] in self.char_to_ix:
                    x = np.zeros(self.vocab_size)
                    x[self.char_to_ix[first_word[t]]] = 1
                    emb_list_1.append(x)
                else:
                    emb_list_1.append(np.zeros(self.vocab_size))
            x_1.append(np.array(emb_list_1))

            for t in range(len(second_word)):
                if second_word[t] in self.char_to_ix:
                    x = np.zeros(self.vocab_size)
                    x[self.char_to_ix[second_word[t]]] = 1
                    emb_list_2.append(x)
                else:
                    emb_list_2.append(np.zeros(self.vocab_size))
            x_2.append(np.array(emb_list_2))

        x_1_pad_seq = keras.preprocessing.sequence.pad_sequences(x_1)
        x_2_pad_seq = keras.preprocessing.sequence.pad_sequences(x_2)
        test = [x_1_pad_seq, x_2_pad_seq]
        self.model.fit([x_1_pad_seq, x_2_pad_seq], targets,
                       batch_size=batch_size, epochs=max_epochs,
                       validation_split=validation_split)

    def fit_gen(self, generator, max_epochs):
        '''    Fit model avec le generateur '''
        self.model.fit_generator(generator, epochs = max_epochs)

    def vectorize(self, words, maxlen_padseq=None):
        '''
        Returns embeddings for list of words. Uses cache of word embeddings to vectorization speed up.
        :param words: list or numpy.ndarray of strings.
        :param maxlen_padseq: parameter 'maxlen' for keras pad_sequences transform.
        :return word_vectors: numpy.ndarray, word embeddings.
        '''
        if not isinstance(words, list) and not isinstance(words, np.ndarray):
            raise TypeError("parameter 'words' must be a list or numpy.ndarray")
        words = [w.lower() for w in words]
        unique_words = np.unique(words)
        new_words = [w for w in unique_words if w not in self.cache]

        if len(new_words) > 0:
            list_of_embeddings = []
            for current_word in new_words:
                if not isinstance(current_word, str):
                    raise TypeError("word must be a string")
                current_embedding = []

                for t in range(len(current_word)):
                    if current_word[t] in self.char_to_ix:
                        x = np.zeros(self.vocab_size)
                        x[self.char_to_ix[current_word[t]]] = 1
                        current_embedding.append(x)
                    else:
                        current_embedding.append(np.zeros(self.vocab_size))

                list_of_embeddings.append(np.array(current_embedding))

            embeddings_pad_seq = keras.preprocessing.sequence.pad_sequences(list_of_embeddings, maxlen=maxlen_padseq)
            new_words_vectors = self.embedding_model.predict([embeddings_pad_seq])
            for i in range(len(new_words)):
                self.cache[new_words[i]] = new_words_vectors[i]
        word_vectors = [self.cache[current_word] for current_word in words]
        return np.array(word_vectors)

    def load_weights(self, path):
        '''
        Loading weights
        Input:  path, folder containing model (must contain weights.h5 and model.pkl)
        '''
        path_to_model = os.path.dirname(os.path.abspath(__file__)) + '/trained_model/' + path
        try:
            self.model.load_weights(path_to_model + '/weights.h5')
            print("\nWeights loaded successfully ")
        except:
            raise
            print("!!!Error loading weights " + path + "!!!\n")
        self.embedding_model.compile(optimizer='adam', loss='mae')

    def save_model(self, path, suffix=''):
        '''
        Saves trained model to directory.
        :param c2v_model: Chars2Vec object, trained model.
        :param path_to_model: str, path to save model.
        '''
        path_to_model = os.path.dirname(os.path.abspath(__file__)) + '/trained_model/' + path

        if not os.path.exists(path_to_model):
            os.makedirs(path_to_model)
        try:
            self.model.save_weights(path_to_model + '/weights' + suffix + '.h5')
            print("\nWeights saved successfully ")
        except:
            print("Error saving weights")
        with open(path_to_model + '/model' + suffix + '.pkl', 'wb') as f:
            pickle.dump([self.dim, self.char_to_ix], f, protocol=2)

class DataGenerator(keras.utils.Sequence):
    """Generate batchs of 128, 10000 batchs by epoch maximum, data is shuffled randomly"""
    def __init__(self, x_set, y_set, char_to_ix, batch_size=256, emb_dim = 50, shuffle=True):
        self.x = x_set
        self.y = y_set
        self.len_data = len(x_set)
        self.batch_size = batch_size
        self.char_to_ix = char_to_ix
        self.vocab_size = len(self.char_to_ix)
        self.dim = emb_dim
        self.shuffle = shuffle
        self.index = 0
        self.on_epoch_end()

    def __len__(self):
        length = int(np.floor(len(self.x) / self.batch_size))
        if length > 10000:
            return 10000
        else:
            return length

    def __getitem__(self, index):
        self.index = random.randint(0, self.len_data - (self.batch_size + 1))
        X = self.x[self.index: self.index + self.batch_size]
        y = self.y[self.index: self.index + self.batch_size]
        x_1, x_2 = [], []
        for pair_words in X:
            emb_list_1, emb_list_2 = [], []
            first_word = pair_words[0].lower()
            second_word = pair_words[1].lower()
            for t in range(len(first_word)):
                if first_word[t] in self.char_to_ix:
                    x = np.zeros(self.vocab_size)
                    x[self.char_to_ix[first_word[t]]] = 1
                    emb_list_1.append(x)
                else:
                    emb_list_1.append(np.zeros(self.vocab_size))
            x_1.append(np.array(emb_list_1))
            for t in range(len(second_word)):
                if second_word[t] in self.char_to_ix:
                    x = np.zeros(self.vocab_size)
                    x[self.char_to_ix[second_word[t]]] = 1
                    emb_list_2.append(x)
                else:
                    emb_list_2.append(np.zeros(self.vocab_size))
            x_2.append(np.array(emb_list_2))
        x_1_pad_seq = keras.preprocessing.sequence.pad_sequences(x_1)
        x_2_pad_seq = keras.preprocessing.sequence.pad_sequences(x_2)
        return [x_1_pad_seq, x_2_pad_seq], y

    def on_epoch_end(self):
        pass
