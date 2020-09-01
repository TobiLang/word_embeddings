'''
Created on 31.08.2020

@author: Tobias Lang
'''

from glove.glove import Glove


def main():
    '''
    Calling GloVe training. No CLI yet.
    '''
    print("Running GloVe...")
    glove = Glove()
    glove.run()


if __name__ == "__main__":
    main()
