import numpy as np
from utils import init_weights

TEST_FILE = 'text8_modified.txt'
VECTOR_SIZE = 10

class Word2Vec() :

    ''' 텍스트 파일의 단어들을 리스트로 바꿔줌 '''
    def read_words(self, words_file):
        with open(words_file, 'r') as f:
            ret = []
            for line in f:
                ret += line.split()

        global WORD_LENGTH # 단어 개수를 전역변수로 선언
        WORD_LENGTH = len(list(set(ret))) # text8_modified.txt : 11233

        return ret

    ''' 단어를 가진 리스트의 원소 하나씩 꺼내서 one-hot vector로 변환 '''
    # TODO 2018-05-11 15:47 각 단어들을 one-hot encoding 하기
    def convert_to_vector(self, word_list):
        for word in word_list:
             print(word)

    ''' 단어를 indexing '''
    def indexing_words(self, word_list):
        word_list = list(set(word_list)) # 중복제거

        indexed_word_dict = {} # 단어별로 indexing된 dictionary 선언

        i = 0
        for word in word_list:
            indexed_word_dict[word] = i
            i += 1

        return indexed_word_dict

    ''' 단어별로 빈도 수를 세어서 dictionary 생성 '''
    def get_frequency_words(self, word_list):

        word_frequency_dict = {}

        # 중복제거된 단어 리스트에서 빈도 수를 세어서 dictionary에 저장
        for word in list(set(word_list)):
            word_frequency_dict[word] = word_list.count(word)

        return word_frequency_dict

    ''' 단어의 index를 받아서 input-hidden weight(in_weight)에서 index의 row 값을 리턴'''
    def input_to_hidden(self, in_weight, word_index):
        return in_weight[word_index]

    def hidden_to_output(self, hidden_vector, out_weight):
        return np.dot(hidden_vector, out_weight)

if __name__ == "__main__":
    word2vec = Word2Vec()
    word_list = word2vec.read_words(TEST_FILE)
    # word2vec.indexing_words(word_list)
    # word2vec.get_frequency_words(word_list)

    # print(len(word_list)) # 17005207개
    # word2vec.convert_to_vector(word_list)
