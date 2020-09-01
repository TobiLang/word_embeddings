# word_embeddings

A Keras implemenation of some Embedding algorithms (word2vec, GloVe).

## Preprocessor

Data preprocessing and normalization used in Word2Vec and GloVe.

## word2vec

This was used by me to get started with Keras & TensorFlow 2.0.
Therefore, this is basically a reprogramming of the [Word2Vec Keras Tutorial](https://adventuresinmachinelearning.com/word2vec-keras-tutorial/).
However, I adjusted
* SkipGrams generation (Negative Sampling size)
* Training (direct training of the Model via .fit())
* Removing the callback to the 2nd model, and calculating the distances between the embeddings using Scikits _euclidean_distances_ function.

I have not implemented CLI switches, but running from the CLI works:

> python3 -m word2vec

## GloVe

This is again a Keras & TensorFlow 2.0 implementation of the [GloVe algorithm](https://nlp.stanford.edu/projects/glove/). The paper nicely describes all
the necessary steps, and I could further explore Keras - as there is the need to implement a custom loss function.

My implemenation of the cooccurrence matrix always uses a symmetric window. And I did not (yet?) spend time on making it fast or memory-friendly (besides using
a Scipy Sparse matrix). In a real word application, this would be a nice task for a Map-Reduce algorithm. Given splitted sentences (the cooccurence matrix in the
paper was calculated per sentence):
* Calculate the Cooccurences for each sentence -> Map.
* Combine them (as it is just a some) to the full matrix -> Reduce.

Running from the CLI:

> python3 -m glove