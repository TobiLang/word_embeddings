'''
Glove Model Generation, Training and validation.

@author: Tobias Lang
'''

import logging

from time import time
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Input, Dot, Embedding, Reshape
import tensorflow.keras.backend as K

from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity


class GloveModel:
    '''
    Prepare, build, train and validate the GloVe model.
    '''

    # Loss Variables (as given by Penninger et. all.)
    X_MAX = 100
    ALPHA = 3. / 4.

    @staticmethod
    def calculate_cooccur(left_side, right_side, target_index, cooccur_matrix):
        '''
        Calculate the cooccurrence for the current target given left and right windows.
        '''
        entries_left = len(left_side)

        for index, entry in enumerate(left_side):
            # Get Distance (reverse numeration)
            distance = entries_left - index
            cooccur_matrix[target_index, entry] += 1. / distance

        for index, entry in enumerate(right_side):
            # Get Distance
            distance = index + 1
            cooccur_matrix[target_index, entry] += 1. / distance

        return cooccur_matrix

    @classmethod
    def create_cooccurence_matrix(cls, normalized_doc, vocabulary_size, window_size=10):
        '''
        Create the co-occurence matrix X_ij: Number of times, j appears in
        the context of i (within a given window size).
        A decreasing weightingfunction is used, so that word pairs that are
        d-words apartcontribute 1/d to the total count.

        normalized_doc: Normalized document nested array of mapped sentences.
        vocabulary_size: Size of the given vocabulary data has been compiled against.
        window_size: Size of the symmetric window to use
        '''

        # Use sparse matrix to store cooccurrences
        cooccur_matrix = lil_matrix((vocabulary_size, vocabulary_size))

        # Loop over sentences
        for sentence in normalized_doc:
            # Loop over data
            for index, target_word in enumerate(sentence):
                # 1. get left n-words
                left_start = max(0, index - window_size)
                left_side = sentence[left_start:index]

                # 2. get right n-words
                right_end = min(len(sentence), index + window_size + 1)
                right_side = sentence[index + 1:right_end]

                # Calculate weighted distances for left/right vs target
                cooccur_matrix = GloveModel.calculate_cooccur(left_side, right_side,
                                                              target_word, cooccur_matrix)

        return cooccur_matrix

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

        # As suggested by the paper, the final weights are the average of w_c.T and w_t:
        final_weights = (model.get_weights()[0] + model.get_weights()[1]) / 2

        # Compute Pairwise Distance matrix: Vocabulary x Vocabulary
        distance_matrix = cosine_similarity(final_weights)

        # Get Top Embeddings for the Valid-Examples
        for i in range(valid_size):
            # Idx->Word
            valid_word = reverse_dictionary[valid_examples[i]]
            # number of nearest neighbors
            top_k = 8
            valid_idx = valid_examples[i]
            # Select Idx Row, and sort columns, cosine similarity increases
            # therefore, the last k-entries have the highest similarity to i.
            # We keep the last entry as a check. It should be the same as i.
            nearest = distance_matrix[valid_idx].argsort()[-top_k:]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            logging.info(log_str)

    @classmethod
    def train_model(cls, model, cooccurrence_matrix, epochs=5):
        '''
        Train the model on a given cooccurence_matrix.
        '''

        # Create Targes, Contexts and Labels
        targets = list()
        contexts = list()
        labels = list()

        # We only train on non-zero elements of X_ij
        # Otherwise, the loss would be -inf due to log(0).
        coo_cooccur = cooccurrence_matrix.tocoo()
        for i, j, xij in zip(coo_cooccur.row, coo_cooccur.col, coo_cooccur.data):
            targets.append(i)
            contexts.append(j)
            labels.append(xij)

        targets = np.asarray(targets)
        contexts = np.asarray(contexts)
        labels = np.asarray(labels)

        # Run Training
        start_time = time()
        loss = model.fit([targets, contexts], labels, epochs=epochs)
        end_time = time()

        logging.info("Training finished: Iterations: %d, loss=%f, Run-Time: %d sec",
                     epochs, loss.history['loss'][-1], int(end_time - start_time))

    @classmethod
    def create_model(cls, vocabulary_size=10000, embedding_dim=300):
        '''
        Create the Glove model:
           w_c.T . w_t + b_c + b_t = log(X_ij)

        vocabulary_size: Size of the given vocabulary data has been compiled against.
        embedding_dim: Desired dimension of the embedding.
        '''

        # Input variables
        target_word = Input((1,), name="input_target_word")
        context_word = Input((1,), name="input_context_word")

        #
        # Would also work with just a DenseLayer(
        #
        # Create Embeddings
        # Keras Embeddings turns indices into dense vectors of fixed size
        # Target Vector and Bias
        target_emb = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim,
                               embeddings_initializer="glorot_uniform",
                               input_length=1, name="target_embedding")
        target_bias = Embedding(input_dim=vocabulary_size, output_dim=1,
                                embeddings_initializer="glorot_uniform",
                                input_length=1, name="target_bias")
        # Context Vector and Bias
        context_emb = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim,
                                embeddings_initializer="glorot_uniform",
                                input_length=1, name="context_embedding")
        context_bias = Embedding(input_dim=vocabulary_size, output_dim=1,
                                 embeddings_initializer="glorot_uniform",
                                 input_length=1, name="context_bias")

        # Calculate Embedding and Bias Vectors. Shapes are for embedding_dim=3
        target_emb_vec = target_emb(target_word)  # S: (,1,3)
        context_emb_vec = context_emb(context_word)  # S: (,1,1)
        target_emb_bias = target_bias(target_word)  # S: (,1,3)
        context_emb_bias = context_bias(context_word)  # S: (,1,1)

        # Dot-Product
        dot_product = Dot(axes=-1)([target_emb_vec, context_emb_vec])  # (,1,3) . (,1,3) -> (,1,1)

        # Reshape to Vectors - Collapse the (1,1) to (1)
        output = Reshape(target_shape=(1,))(dot_product)  # (,1,1) -> (,1)
        target_bias = Reshape(target_shape=(1,))(target_emb_bias)  # (,1,1) -> (,1)
        context_bias = Reshape(target_shape=(1,))(context_emb_bias)  # (,1,1) -> (,1)
        # Add outputs
        output = Add()([output, target_bias, context_bias])

        # Create and compile Model
        model = Model(inputs=[target_word, context_word], outputs=output,
                      name="glove_model")
        model.run_eagerly = True
        # Optimizer: Paper used AdaGrad, could also be Adam, ...
        model.compile(optimizer="adam", loss=GloveModel.custom_loss)

        return model

    @classmethod
    def custom_loss(cls, y_true, y_pred):
        '''
        The GloVe loss function is given as:
           SUM_ij = f(X_ij) * (y_pred - y_true)^2

           With y_pred = w_c.T . w_t + b_c + b_t
                y_true = log(X_ij)

           And f(X_ij) = (x/x_max)^alpha if (x < x_max)
                         1 if (x >= x_max)

        '''
        # Calculate squared distance:
        squared = K.square(y_pred - K.log(y_true))
        # Calculate weighting function
        #
        # #f_xij = K.pow((tf.minimum(y_true, cls.X_MAX) / cls.X_MAX), cls.ALPHA)
        #
        # I find K.clip more elegant, as we are able to define a min and a max
        weighting = K.pow(K.clip(y_true / cls.X_MAX, 0.0, 1.0), cls.ALPHA)

        return K.sum(weighting * squared, axis=-1)
