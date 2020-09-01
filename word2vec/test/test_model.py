'''
Model Unit-tests.

@author: Tobias Lang
'''

import os
import unittest

import numpy as np
from word2vec.model import Word2VecModel


class TestModel(unittest.TestCase):
    '''
    Testing of model generation, and skip-grams preparation.
    '''
    RESOURCE_BASE_PATH = ('../data')
    FILE_PATH = os.path.join(RESOURCE_BASE_PATH, 'zarathustra.txt')

    def test_skipgrams(self):
        '''
        Testing generation of skip-grams to be used as model input/output parameters.
        '''
        vocabulary_size = 2000
        data = [[1376, 1, 1377, 1378, 412, 1379, 788, 1380, 788, 42, 329],
                [789, 191, 111, 112, 138, 575, 162]]

        ratio = 3.0
        target, context, label = Word2VecModel.create_skipgrams(data, vocabulary_size, ratio)

        self.assertEqual(target.shape, context.shape)
        pos_labels = np.sum(np.array(label) == 1)
        neg_labels = np.sum(np.array(label) == 0)
        self.assertEqual(neg_labels / pos_labels, ratio)

    @unittest.skip("Deactivated, time-consuming, and just a test to see the model summary.")
    @classmethod
    def test_model(cls):
        '''
        Check whether the model compiles or not.
        '''
        model = Word2VecModel.create_model(10000, 300)
        print(model.summary())


if __name__ == '__main__':
    unittest.main()
