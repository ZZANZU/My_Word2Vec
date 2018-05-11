import numpy as np

class Word2Vec() :

    """ 텍스트 파일의 단어들을 리스트로 바꿔줌 """
    # TODO 2018-05-11 15:37 중복 단어 제거
    def read_words(self, words_file):
        with open(words_file, 'r') as f:
            ret = []
            for line in f:
                ret += line.split()
            return ret


if __name__ == "__main__":
    word2vec = Word2Vec()

    print(word2vec.read_words('text8.txt'))