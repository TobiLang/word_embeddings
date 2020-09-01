'''
Check Normalization and preprocessing.

@author: Tobias Lang
'''
import unittest

import os
from nltk.corpus import gutenberg
from preprocessor.text_preprocessor import TextPreprocessor


class TestTextPreprocessing(unittest.TestCase):
    '''
    Check Normalization and PreProcessing.
    '''

    RESOURCE_BASE_PATH = ('../../data')
    FILE_PATH = os.path.join(RESOURCE_BASE_PATH, 'zarathustra.txt')

    def test_read_file(self):
        '''
        Testing File reading.
        '''
        expected = ('Ein Buch\n'
                    'für\n'
                    'Alle und Keinen')
        doc = TextPreprocessor.read_file(self.FILE_PATH)

        self.assertEqual(len(doc), 120483)
        self.assertEqual(doc[92:120], expected)

    def test_normalize_german(self):
        '''
        Testing normalization of german sentences.
        '''
        input_doc = ("Felix Heuberger (* 7. März 1888 in Wien; † 25. Jänner 1968 in Hall "
                     "in Tirol) war ein österreichischer Maler, Radierer und Ingenieur. "
                     "Die Pfarrkirche wurde 1066 als „ecclesia Grazluppa“ erstmals urkundlich "
                     "erwähnt. "
                     "Spanisch ​[⁠β⁠]​: jedes b und v, das nicht im absoluten Anlaut steht und "
                     "auch einem Nasal folgt. ")

        expected_doc = ("felix heuberger märz wien jänner hall tirol österreichischer "
                        "maler radierer ingenieur "
                        "pfarrkirche wurde ecclesia grazluppa erstmals urkundlich erwähnt "
                        "spanisch b v absoluten anlaut steht nasal folgt")

        normalized_doc = TextPreprocessor.normalize(input_doc, "german")

        self.assertTrue(normalized_doc == expected_doc)

    def test_normalize_english(self):
        '''
        Testing normalization of english sentences.
        '''
        input_doc = ("The symbol ⟨β⟩ is the Greek letter beta. "
                     "Song, J.; Choo, Y. -J.; Cho, J. -C. (2008). \"Perlucidibaca piscinae gen. "
                     "nov., sp. nov., a freshwater bacterium belonging to the family "
                     "Moraxellaceae\" "
                     "The leaves range from 2 to 12 centimeters (0.79 to 4.72 in) in length and "
                     "1 to 5 centimeters (0.39 to 1.97 in) in breadth. ")

        expected_doc = ("symbol greek letter beta "
                        "song j choo j cho j c perlucidibaca piscinae gen nov sp nov freshwater "
                        "bacterium belonging family moraxellaceae "
                        "leaves range centimeters length centimeters breadth")

        normalized_doc = TextPreprocessor.normalize(input_doc, "english")
        self.assertTrue(normalized_doc == expected_doc)

    def test_preprocess_king_james_bible(self):
        '''
        Testing Preprocessing on the KingJamesBible.
        '''
        # Load King James Bible
        bible = gutenberg.sents('bible-kjv.txt')

        # Expected
        expected_one = ['old', 'testament', 'king', 'james', 'bible']
        expected_ten = ['god', 'said', 'let', 'firmament', 'midst', 'waters', 'let',
                        'divide', 'waters', 'waters']

        norm_bible = TextPreprocessor.preprocess(bible[:11], "english")

        self.assertEqual(norm_bible[1], expected_one)
        self.assertEqual(norm_bible[10], expected_ten)

    def test_preprocess_nietzsche_zarathustra(self):
        '''
        Testing Preprocessing on Zarathustra.
        '''
        # Set Zarathustra (not in NLTK corpus)
        zarathustra = ('[5/0011] '
                       'I. '
                       'Als Zarathustra dreissig Jahr alt war, verliess er '
                       'seine Heimat und den See seiner Heimat und gieng in '
                       'das Gebirge. Hier genoss er seines Geistes und seiner '
                       'Einsamkeit und wurde dessen zehn Jahre nicht müde. '
                       'Endlich aber verwandelte sich sein Herz, — und eines '
                       'Morgens stand er mit der Morgenröthe auf, trat vor '
                       'die Sonne hin und sprach zu ihr also:')

        # Expected
        expected_zero = ['i', 'zarathustra', 'dreissig', 'jahr', 'alt', 'verliess', 'heimat',
                         'see', 'heimat', 'gieng', 'gebirge']
        expected_two = ['genoss', 'geistes', 'einsamkeit', 'wurde', 'zehn', 'jahre', 'müde']

        norm_zarathustra = TextPreprocessor.preprocess(zarathustra, "german")

        self.assertEqual(norm_zarathustra[0], expected_zero)
        self.assertEqual(norm_zarathustra[1], expected_two)

    def test_preprocess_nietzsche_zarathustra_from_file(self):
        '''
        Testing Preprocessing on Zarathustra from a file.
        '''
        # Set Zarathustra (not in NLTK corpus)
        zarathustra = TextPreprocessor.read_file(self.FILE_PATH)

        # Expected
        expected_nineteen = ['i', 'zarathustra', 'dreissig', 'jahr', 'alt', 'verliess',
                             'heimat', 'see', 'heimat', 'gieng', 'gebirge']
        expected_twenty = ['genoss', 'geistes', 'einsamkeit', 'wurde', 'zehn', 'jahre', 'müde']

        norm_zarathustra = TextPreprocessor.preprocess(zarathustra, "german")

        self.assertEqual(norm_zarathustra[19], expected_nineteen)
        self.assertEqual(norm_zarathustra[20], expected_twenty)

    def test_convert_to_dictionary(self):
        '''
        Testing convertion of a normalized doc to a dictionary and mapped dataset.
        '''
        # Setup small norm_doc
        norm_doc = [['old', 'testament', 'king', 'james', 'bible'],
                    ['god', 'said', 'let', 'firmament', 'midst', 'waters', 'let',
                     'divide', 'waters', 'waters']]

        expected_dictionary = {'<unk>': 0, 'bible': 7, 'divide': 12, 'firmament': 10,
                               'god': 8, 'james': 6, 'king': 5, 'let': 2, 'midst': 11,
                               'old': 3, 'said': 9, 'testament': 4, 'waters': 1}
        expected_reverse_dictionary = {0: '<unk>', 1: 'waters', 2: 'let', 3: 'old', 4:
                                       'testament', 5: 'king', 6: 'james', 7: 'bible', 8:'god',
                                       9: 'said', 10: 'firmament', 11: 'midst', 12: 'divide'}
        expected_data = [[3, 4, 5, 6, 7],
                         [8, 9, 2, 10, 11, 1, 2, 12, 1, 1]]

        data, dictionary, reverse_dictionary = TextPreprocessor.convert_to_dictionary(norm_doc, 100)

        for i in range(12):
            word = reverse_dictionary[i]
            idx = dictionary[word]
            self.assertEqual(i, idx)

        self.assertEqual(data, expected_data)
        self.assertEqual(dictionary, expected_dictionary)
        self.assertEqual(reverse_dictionary, expected_reverse_dictionary)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
