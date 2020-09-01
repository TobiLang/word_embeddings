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
        data = [30, 1542, 31, 30, 1543, 31, 30, 1544, 31, 30, 1545, 31, 30, 1546, 31,
                30, 1547, 31, 30, 30, 1548, 31, 1549, 31, 46, 61, 29, 2, 17, 1550, 82,
                87, 3, 381, 2, 33, 1551, 1552, 2, 1553, 1554, 1555, 33, 976, 1556, 2,
                1557, 1558, 1559, 977, 1560, 1561, 66, 2, 1562, 1563, 1564, 1565, 978,
                1566, 2, 1567, 2, 979, 1568, 2, 1569, 2, 584, 1570, 1571, 2, 1572, 978,
                1573, 1, 1574, 2, 979, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582,
                1583, 980, 1584, 2, 1585, 1586, 1587, 2, 1588, 1589, 980, 1590, 1591]

        ratio = 3.0
        dummy, dummy2, label = Word2VecModel.create_skipgrams(data, vocabulary_size, ratio)

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
