'''
Word2Vec Unit-tests.

@author: Tobias Lang
'''

import os
import unittest

from word2vec.word2vec import Word2Vec


class TestWord2Vec(unittest.TestCase):
    '''
    As testing the running of the model would take to long,
    and be an application test, we just check the data-preparation
    pipeline.
    '''
    RESOURCE_BASE_PATH = ('../../data')
    FILE_PATH = os.path.join(RESOURCE_BASE_PATH, 'zarathustra.txt')

    def test_prepare_data(self):
        '''
        Validate data preparation and preprocessing.
        '''
        mapped_data, dummy, reverse_dict = Word2Vec.prepare_data(self.FILE_PATH, "german", 3500)

        translated = list()
        for idx in mapped_data[19]:
            translated.append(reverse_dict[idx])

        # Check some sentence
        self.assertEqual(mapped_data[19], [1376, 1, 1377, 1378, 412, 1379,
                                           788, 1380, 788, 42, 329])
        self.assertEqual(translated, ['i', 'zarathustra', 'dreissig', 'jahr', 'alt', 'verliess',
                                      'heimat', 'see', 'heimat', 'gieng', 'gebirge'])


if __name__ == '__main__':
    unittest.main()
