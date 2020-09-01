'''
Word2Vev SkipGram and Model Generation, Training and validation.

See: https://adventuresinmachinelearning.com/word2vec-keras-tutorial/

@author: Tobias Lang
'''

from time import time
import logging
import itertools

from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Reshape, Dot, Dense
from tensorflow.keras.models import Sequential

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


class Word2VecModel:
    '''
    Prepare, build, train and validate the Word2Vec model.
    '''

    @classmethod
    def create_skipgrams(cls, normalized_doc, vocabulary_size=10000, ratio=3.0):
        '''
        Create the skipgrams to be trained on the model

        normalized_doc: Normalized document nested array of mapped sentences.
        vocabulary_size: Size of the given vocabulary data has been compiled against.
        ratio: Negative to Positive sampling ratio.
        '''
        # Used for generating the sampling_table argument for skipgrams. sampling_table[i] is the
        # probability of sampling the word i-th most common word in a dataset (more common words
        # should be sampled less frequently, for balance).
        sampling_table = sequence.make_sampling_table(vocabulary_size)

        # Create Skipgrams with Keras
        # This function transforms a sequence of word indexes (list of integers) into tuples of
        # words of the form:
        #   * (word, word in the same window), with label 1 (positive samples).
        #   * (word, random word from the vocabulary), with label 0 (negative samples).
        # Flatten normalized document
        data = list(itertools.chain.from_iterable(normalized_doc))
        couples, labels = skipgrams(data, vocabulary_size,
                                    negative_samples=ratio,
                                    sampling_table=sampling_table)

        # Split couples into target and context
        word_target, word_context = zip(*couples)

        # Convert to Numpy array, ! rank 1 array!
        word_target = np.array(word_target, dtype="int32")
        word_context = np.array(word_context, dtype="int32")

        return word_target, word_context, labels

    @classmethod
    def create_model(cls, vocabulary_size=10000, embedding_dim=300):
        '''
        Create the Word2Vec model

        vocabulary_size: Size of the given vocabulary data has been compiled against.
        embedding_dim: Desired dimension of the embedding.
        '''

        # Create the target model
        target_model = Sequential()
        target_model.add(Embedding(vocabulary_size, embedding_dim,
                                   embeddings_initializer="glorot_uniform",
                                   input_length=1))
        target_model.add(Reshape((embedding_dim,)))

        # Create the context model
        context_model = Sequential()
        context_model.add(Embedding(vocabulary_size, embedding_dim,
                                    embeddings_initializer="glorot_uniform",
                                    input_length=1))
        context_model.add(Reshape((embedding_dim,)))

        # Now perform the dot product operation to get a similarity measure
        dot_product = Dot(axes=1, normalize=False)([target_model.output, context_model.output])

        # Add the sigmoid output layer - costly operation, though
        dot_product = Dense(1, kernel_initializer="glorot_uniform",
                            activation="sigmoid")(dot_product)

        # Create the Model
        model = Model(inputs=[target_model.input, context_model.input],
                      outputs=dot_product, name="word2vec_model")

        # Optimizer: Adam or RMSProb, loss: BinaryCrossEntropy 0/1 labels
        model.compile(loss="binary_crossentropy", optimizer="adam")

        return model

    @classmethod
    def train_model(cls, model, word_target, word_context, word_labels, epochs=2):
        '''
        Train the Word2Vec Model on the given Inputs:
          * word_target
          * word_context
        and Output
          * labels

        * Train on
        '''
        # Create Input/Output arrays (1,0)
        input_target = np.asarray(word_target)
        input_context = np.asarray(word_context)
        output_label = np.asarray(word_labels)

        # Run Training
        start_time = time()
        loss = model.fit([input_target, input_context], output_label,
                         batch_size=32, epochs=epochs)
        end_time = time()

        logging.info("Training finished: Iterations: %d, loss=%f, Run-Time: %d sec",
                     epochs, loss.history['loss'][-1], int(end_time - start_time))

    @classmethod
    def test_similarity_model(cls, model, dictionary, reverse_dictionary):
        '''
        Check the similarity of a given list of words to test the quality of the embedding.
        '''
        # For Bible:
        valid_examples = list()
        for entry in ['god', 'jesus', 'noah', 'egypt', 'john', 'gospel', 'moses', 'famine']:
            if dictionary.get(entry, None):
                valid_examples.append(dictionary[entry])
        valid_size = len(valid_examples)

        # Extract Model Weights (Embedding-Matrix - EmbeddingDim x Vocabulary)
        weights = model.get_weights()[0]
        weights = weights[1:]

        # Compute Pairwise Distance matrix: Vocabulary x Vocabulary
        distance_matrix = cosine_similarity(weights)

        # Get Top Embeddings for the Valid-Examples
        for i in range(valid_size):
            # Idx->Word
            valid_word = reverse_dictionary[valid_examples[i]]
            # number of nearest neighbors
            top_k = 8
            valid_idx = valid_examples[i]
            # Select Idx Row, and sort columns per maximum
            nearest = distance_matrix[valid_idx].argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            logging.info(log_str)
