import numpy as np
from utils import init_weights, softmax, one_hot

# text8 : # 17005207개
TEST_FILE = 'text8_modified.txt'
EMBEDDING_SIZE = 10
LEARNING_RATE = 0.1

class Word2Vec() :

    ''' 텍스트 파일의 단어들을 리스트로 바꿔줌 '''
    def read_words(self, words_file):
        with open(words_file, 'r') as f:
            input_words = []
            for line in f:
                input_line = line.split()
                for word in input_line:
                    input_words.append(word)

        global WORD_LENGTH # 단어 개수를 전역변수로 선언
        WORD_LENGTH = len(input_words)

        return input_words

    ''' 단어를 가진 리스트의 원소 하나씩 꺼내서 one-hot vector로 변환 '''
    # TODO 2018-05-11 15:47 각 단어들을 one-hot encoding 하기
    def convert_to_vector(self, word_list):
        for word in word_list:
             print(word)

    ''' 단어를 indexing '''
    def indexing_words(self, word_list):
        # word_list = list(set(word_list)) # 중복제거

        indexed_word_dict = {} # 단어별로 indexing된 dictionary 선언

        i = 0
        for word in word_list:
            indexed_word_dict[i] = word # key : 0~ , vlaue : 단어
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
        u = np.dot(out_weight.T, hidden_vector) # W'T (V x N) * h (N x 1)

        return u

    def skip_gram(self, learning_rate, window_size):
        word2vec = Word2Vec()
        word_list = word2vec.read_words(TEST_FILE) # 단어 리스트 생성 & 단어 개수 파악

        indexed_word = word2vec.indexing_words(word_list) # 단어에 indexing

        in_weight = init_weights((WORD_LENGTH, EMBEDDING_SIZE)) # W 생성
        out_weight = init_weights((EMBEDDING_SIZE, WORD_LENGTH)) # W' 생성

        loss = 0

        for i in range(0, WORD_LENGTH + 1):
            if i % 100 == 0:
                print("learning word [" + str(i) + "]" + indexed_word.get(i))
                print("loss : " + loss)

            center_one_hot = one_hot(word_list, i)
            context = [] # center word 주변 단어들을 저장

            ''' context 리스트 생성 '''
            for j in range(i - window_size, i + window_size + 1): # 현재 단어에서 window size 반경의 단어
                if j >= 0 and j != i: # 단어의 index가 0 이상이고 center word가 아닌 것들 중에서만
                    context.append(indexed_word[j]) # context word에 추가

            ''' 실제 연산 & weight update'''
            for j in range(i - window_size, i + window_size + 1):
                if j >= 0 and j != i:
                    h = word2vec.input_to_hidden(in_weight, i)  # hidden layer의 vector 생성, 실제로는 곱하지 않고 단어의 index 위치의 row를 읽어온다
                    u = word2vec.hidden_to_output(h, out_weight)  # hidden layer의 vector와 W'을 곱하고 softmax를 한다.
                    y = softmax(u)

                    label_one_hot = one_hot(word_list, i+j)

                    e = np.array([-label_one_hot + y.T for label in context])
                    new_in_weight = np.outer(center_one_hot, np.dot(out_weight, np.sum(e, axis=0)))
                    new_out_weight = np.outer(h, np.sum(e, axis=0))

                    in_weight = in_weight - learning_rate * new_in_weight
                    out_weight = out_weight - learning_rate * new_out_weight

                    loss += -np.sum([u[label_one_hot == 1] for label in context]) + len(context) * np.log(np.sum(np.exp(u)))


if __name__ == "__main__":
    word2vec = Word2Vec()
    word2vec.skip_gram(learning_rate=0.1, window_size=2)