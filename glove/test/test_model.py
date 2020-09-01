'''
Created on 31.08.2020

@author: Tobias Lang
'''
import unittest

from collections import defaultdict
import tensorflow as tf
import numpy as np

from glove.model import GloveModel


class TestModel(unittest.TestCase):
    '''
    Testing of model generation, and cooccurrence matrix generation.
    '''

    def setUp(self):
        np.random.seed(42)
        tf.random.set_seed(42)

        self.vocab_size = 10
        self.emb_dim = 3

    def test_count_cooccur(self):
        '''
        Validate correct cooccurrence count for one target given left,right windows.
        '''

        # Use defaultdict .. way more performant in updating elements than a sparse matrix
        cooccurence_matrix = defaultdict(lambda: defaultdict(int))
        sentence = [30, 1542, 31, 30, 1543, 31, 30]

        expected = [(30, 30, 1. / 3. + 1. / 3. + 1. / 3. + 1. / 3.),
                    (30, 1542, 1. / 1. + 1. / 2.),
                    (30, 31, 1. / 2. + 1. / 1. + 1. / 2. + 1. / 1.),
                    (30, 1543, 1. / 1. + 1. / 2.)]

        dummy = GloveModel.calculate_cooccur(sentence, cooccur_matrix=cooccurence_matrix,
                                             window_size=3)

        for i, j, xij in expected:
            self.assertEqual(cooccurence_matrix[i][j], xij)

    def test_create_cooccurence_matrix(self):
        '''
        Validate correct creation of the full cooccurrence matrix given an input (sentences).
        '''
        window_size = 3
        normalized_doc = [[3, 4, 5, 6, 7],
                          [8, 9, 3, 10, 5, 1, 2, 12, 1, 1]]

        cooccurence_matrix = GloveModel.create_cooccurence_matrix(normalized_doc, window_size)

        self.assertEqual(cooccurence_matrix[5][3], 1. / 2. + 1. / 2.)
        self.assertEqual(cooccurence_matrix[1][5], 1. / 1.)
        self.assertEqual(cooccurence_matrix[1][12], 1. / 2. + 1. / 1. + 1. / 2.)

    @unittest.skip("Deactivated, just a first test, whether nor not the model is actually working")
    def test_train_on_batch(self):
        '''
        Check whether the compiled model is actually trainable (especially with the custom loss).
        '''
        model = GloveModel.create_model(self.vocab_size, self.emb_dim)

        cooccurence_matrix = np.random.rand(self.vocab_size, self.vocab_size)
        arr_1 = np.zeros((1,))
        arr_2 = np.zeros((1,))
        arr_3 = np.zeros((1,))
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                arr_1[0, ] = i
                arr_2[0, ] = j
                arr_3[0, ] = cooccurence_matrix[i, j]

                loss = model.train_on_batch([arr_1, arr_2], arr_3)
        print("Iteration {}, loss={}".format(1, loss))
        self.assertAlmostEqual(loss, 0.04529602453112602)

    @unittest.skip("Deactivated, just a first test, whether nor not the model is actually working")
    def test_train_fit(self):
        '''
        Train the model on random data using the .fit() method.
        Will differ from train_on_batch() due to random shuffeling of the batches.
        '''
        model = GloveModel.create_model(self.vocab_size, self.emb_dim)

        cooccurence_matrix = np.random.rand(self.vocab_size, self.vocab_size)

        # Create Input/Output arrays (1,0)
        targets = list()
        contexts = list()
        labels = list()
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                targets.append(i)
                contexts.append(j)
                labels.append(cooccurence_matrix[i, j])
        arr_1 = np.asarray(targets)
        arr_2 = np.asarray(contexts)
        arr_3 = np.asarray(labels)

        loss = model.fit([arr_1, arr_2], arr_3)
        print("Iteration {}, loss={}".format(1, loss.history['loss'][-1]))
        self.assertAlmostEqual(loss.history['loss'][-1], 0.013527743518352509)

    @unittest.skip("Deactivated, time-consuming, and just a test to see the model summary.")
    def test_model(self):
        '''
        Check whether the model compiles or not.
        '''
        model = GloveModel.create_model(self.vocab_size, self.emb_dim)
        print(model.summary())


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
