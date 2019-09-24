# chars2vec

### Character embedding model

chars2vec is a model that enables character-level embedding of words.
It has been developped with the goal of easing the analysis of noisy text coming from user input.

The main use case is to retrieve words with a similar spelling to the input, which may or may not be in the vocabulary.
This enables to efficiently gather a list of candidates for error correction even in a large dictionnary.
This is achieved by vectorizing the input word and searching through the indexed vectorized dictionnary using [nmslib](https://github.com/nmslib/nmslib) to perfrom ANN search ([approximate nearest neighbor](https://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximation_methods))

The model has been trained on 10 millions pairs of words with a binary classifcation objective of determining if words pairs are similars (vocabulary word and same word with one typing mistake) or different (two randomly chosen words)

Embedding size may vary depending on the required accuracy, for our usage we have settled for 150 dimensions. The corresponding model trained on a french dictionnary is provided.

Three files are provided for basic usage (visualization, training and error correction)

### Usage

#### Visualization

`python usage_visualization.py`

Embedding visualization by reducing dimensionality through PCA with the following vocabulary: <br/>
['est', 'ezt', 'zest', <br/>
'carotte', 'carote', 'carottte', <br/>
'langage', 'language', 'langqge', <br/>
'francais', 'franssais', 'francqis']
<br/><kbd>
<img src="https://cdn-images-1.medium.com/max/800/1*0iyZd-0CUAZliw1z2Eoe4Q.png" width='400' height='300'/>
</kbd>

#### Training

`python usage_training.py`

Training data is provided in chars2vec/training_data/training_data.json (1.5 millions word pairs)
This data has been generated using a 600K words dictionnary and a custom typing-error generator.

Modify 'model_chars' if you want additional characters to be taken into account by the model.
By default the model uses a generator in order to reduce memory usage, and saves weights at each epoch.

Due to the nature of the model it is not obvious to determine when to stop training the model.
We have used an empirical approach and have stopped training when the performance peaked for our particular usage.

#### Word correction

`python usage_correction.py`

First usage will take around 3 min depending on your hardware because the model has to build a vectorization of your entire dictionnary and an index for efficient research.
Following usages should be in ms after tensorflow initialization.

Example: <br/>
Edit distance 1: <br/>
langage ['langages', 'langage']<br/>
langqge ['langage']<br/>
langagee ['langages', 'langage']<br/>

EEdit distance 2:<br/>
langage ['langages', 'lange', 'largages', 'tangages', 'langage', 'langé']<br/>
langqge ['langages', 'lange', 'langé', 'langage']<br/>
langagee ['langages', 'largages', 'langage', 'tangages']<br/>

The find_knn function may take a number of knn to search through (default 80) and an accepted [levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) for candidates.
The higher the k variable the more chances you catch the inteded candidate for correction however this will also increase the computation time

### Installation

Download sources and launch with command line:

~~~shell
`python setup.py install`
~~~

### Credits

The original model has been developped by Intuition Engineering.
