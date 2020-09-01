'''
Word 2 Vec in Keras

See: https://adventuresinmachinelearning.com/word2vec-keras-tutorial/

@author: Tobias Lang
'''

import logging

from tensorflow.keras.models import load_model
from preprocessor.text_preprocessor import TextPreprocessor

from .model import GloveModel


class Glove:
    '''
    Run the GloVe algorithm on a given input file.
    '''

    def __init__(self):
        # Set Logging to info
        logging_format = '%(asctime)-15s %(message)s'
        logging.basicConfig(level=logging.INFO, format=logging_format)

    @classmethod
    def prepare_data(cls, input_file=None, language="english", vocabulary_size=10000):
        '''
        Preprocess a given input file, and convert it to a mapped dictionary.
        '''
        # Load data from disk
        document = TextPreprocessor.read_file(input_file)

        # Preprocess and normalize
        norm_doc = TextPreprocessor.preprocess(document, language)

        # Generate mapping, dict and reverse_dict
        return TextPreprocessor.convert_to_dictionary(norm_doc, vocabulary_size)

    def run(self):
        '''
        Run GloVe

        Current-Hyperparameters:
            * Vocabulary size
            * Window size
            * Epochs
            * Embed_Dim
        '''

        data_sets = [("data/the-king-james-bible.txt", "english", 10000, 10, 200, 5)]

        for input_file, language, vocab_size, window_size, embedding_dim, epochs in data_sets:
            # Prepare Data and Vocabulary
            logging.info("Preprocessing data ...")
            normalized_doc, dictionary, reverse_dictionary = self.prepare_data(input_file,
                                                                               language,
                                                                               vocab_size)

            # Generate SkipGrams - Put this inside the Training loop
            logging.info("Building cooccurence matrix ...")
            cooccurence_matrix = GloveModel.create_cooccurence_matrix(normalized_doc,
                                                                      vocab_size, window_size)

            # Generate Model
            model = GloveModel.create_model(vocabulary_size=vocab_size,
                                            embedding_dim=embedding_dim)

            # Train Model
            logging.info("Training model for %d epochs ...", epochs)
            GloveModel.train_model(model, cooccurence_matrix, epochs=epochs)

            # Test Similarity
            logging.info("Checking similarity ...")
            GloveModel.test_similarity_model(model, dictionary, reverse_dictionary)

            # Save Model
            logging.info("Saving model ...")
            model.save("models/glove/{}-{}-{}-{}".format(input_file, vocab_size,
                                                         embedding_dim, epochs))

    def run_eval(self):
        '''
        Load a trained model and run the similiarity check on it.
        '''

        data_sets = [("data/the-king-james-bible.txt", "english", 10000, 10, 200, 5)]

        for input_file, language, vocab_size, dummy, embedding_dim, epochs in data_sets:
            # Prepare Data and Vocabulary
            logging.info("Preprocessing data ...")
            dummy, dictionary, reverse_dictionary = self.prepare_data(input_file,
                                                                      language,
                                                                      vocab_size)

            # Load Model
            logging.info("Loading model ...")
            file_name = "models/glove/{}-{}-{}-{}".format(input_file, vocab_size,
                                                          embedding_dim, epochs)
            # Need to hand over custom_loss
            model = load_model(file_name, custom_objects={'custom_loss': GloveModel.custom_loss})

            # Test Similarity
            logging.info("Checking similarity ...")
            GloveModel.test_similarity_model(model, dictionary, reverse_dictionary)
