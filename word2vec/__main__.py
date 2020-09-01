'''
Created on 31.08.2020

@author: tobias
'''

from word2vec.word2vec import Word2Vec


def main():
    print("Running Word2Vec...")
    w2v = Word2Vec()
    w2v.run()
    # w2v.run_eval()


if __name__ == "__main__":
    main()
