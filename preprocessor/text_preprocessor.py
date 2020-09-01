'''
Simple Text preprocessor to prepare inputs for Word2Vec and GloVe.

@author: Tobias Lang
'''

import sys
import logging
import itertools
from collections import Counter
import regex as re
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords

import numpy as np


class TextPreprocessor:
    '''
    Text Preprocessing handles these steps:

      * Normalization:
        * Remove special chars and lrstrip whitespaces.
        * Convert to lowercase.
        * Tokenize dokument and remove stopwords (using NLTK WordPunctTokenizer)

      * Preprocessing:
        *
    '''

    UNK = '<unk>'

    NON_ALPHA_GER = re.compile(r'[^a-zA-ZäöüÄÖÜß\s]', re.IGNORECASE | re.ASCII)
    NON_ALPHA_ENG = re.compile(r'[^a-zA-Z\s]', re.IGNORECASE | re.ASCII)

    @classmethod
    def read_file(cls, file_path):
        '''
        Read a text document from file.
        '''

        doc = None
        try:
            file_object = open(file_path, "r")
            doc = file_object.read()
            file_object.close()
        except OSError:
            logging.error("Could not open/read file: %s", file_path)
            sys.exit()

        return doc

    @classmethod
    def normalize(cls, input_doc, language="english"):
        '''
        Normalize given input.
        '''

        # Remove special-chars
        if language == "german":
            processed_doc = re.sub(cls.NON_ALPHA_GER, '', input_doc)
        else:
            processed_doc = re.sub(cls.NON_ALPHA_GER, '', input_doc)
        # To Lowercase and Strip Whitespaces
        processed_doc = processed_doc.lower().strip()

        # Tokenize
        tokenizer = WordPunctTokenizer()
        tokens = tokenizer.tokenize(processed_doc)

        # Remove Stopwords.
        if language == "german":
            stop = stopwords.words("german")
            cleaned_tokens = [token for token in tokens if token not in stop]
        else:
            stop = stopwords.words("english")
            cleaned_tokens = [token for token in tokens if token not in stop]

        processed_doc = ' '.join(cleaned_tokens)

        return processed_doc

    @classmethod
    def preprocess(cls, input_doc, language="english"):
        '''
        Preprocess a given input.
        '''

        # Is the input_doc a nested list (e.g NLTK Bible)?
        if isinstance(input_doc, list):
            if isinstance(input_doc[0], list):
                input_sentences = [' '.join(sentence) for sentence in input_doc]
            else:
                # A simple list
                input_sentences = [' '.join(word) for word in input_doc]
        else:
            # A plain string, pre-tokenize to sentences
            sent_tokenizer = PunktSentenceTokenizer("language")
            input_sentences = sent_tokenizer.tokenize(input_doc)

        # Vectorize normalize step
        normalize = np.vectorize(cls.normalize)
        norm_corpus = filter(None, normalize(input_sentences, language))

        # Convert to nested list, split each sentence into a list
        normalized_doc = [sentence.split(' ') for sentence in norm_corpus]

        return normalized_doc

    @classmethod
    def convert_to_dictionary(cls, normalized_doc, vocab_size=10000):
        '''
        Given an normalized_doc (list of sentences), generate two dictionaries:
          * Dictionary: String -> Integer (position) mapping
          * Reverse Dictionary: Integer (position) -> String mapping

        With this, we map the given normalized_doc.
        '''

        # Flatten normalized document
        tokens = list(itertools.chain.from_iterable(normalized_doc))
        # Extract most common tokens
        # Add the UNK token (count -1)
        common_tokens = [[cls.UNK, -1]]
        common_tokens.extend(Counter(tokens).most_common(vocab_size - 1))

        # Put to dictionary and reverse dictionary
        dictionary = dict()
        reverse_dictionary = dict()
        for token, dummy in common_tokens:
            # Use len(dict) as a i += 1 shortcut
            idx = len(dictionary)
            dictionary[token] = idx
            reverse_dictionary[idx] = token

        # Now, build the dataset (word->int)
        mapped_document = list()
        for sentence in normalized_doc:
            mapped_sentence = list()
            for word in sentence:
                if dictionary.get(word, None):
                    index = dictionary[word]
                else:
                    index = dictionary[cls.UNK]
                mapped_sentence.append(index)
            mapped_document.append(mapped_sentence)

        return mapped_document, dictionary, reverse_dictionary
