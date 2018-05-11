import numpy as np

class Word2Vec() :

    """ 텍스트 파일의 단어들을 리스트로 바꿔줌 """
    def read_words(self, words_file):
        with open(words_file, 'r') as f:
            ret = []
            for line in f:
                ret += line.split()

            ret = list(set(ret)) # DONE 2018-05-11 15:37 중복 단어 제거

            return ret

    """ 단어를 가진 리스트의 원소 하나씩 꺼내서 one-hot vector로 변환 """
    # TODO 2018-05-11 15:47 각 단어들을 one-hot encoding 하기
    def convert_to_vector(self):



if __name__ == "__main__":
    word2vec = Word2Vec()

    print(word2vec.read_words('text8.txt'))