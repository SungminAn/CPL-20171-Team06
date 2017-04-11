#-*- coding: utf-8 -*-
import numpy as np
import re
import itertools
import konlpy
from collections import Counter
from tensorflow.contrib import learn
import jpype
hannanum = konlpy.tag.Hannanum()
kkma = konlpy.tag.Kkma() #쓸거면 바꾸기 
twitter = konlpy.tag.Twitter()  #형태소로 나누고 품사 태깅까지

morph = twitter

def clean_str(string, _morph=False): #형태소 분석
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # string = re.sub('[^ \ta-zA-Zㄱ-ㅣ가-힣]+', '', string)
    string = re.sub('[^ \tㄱ-ㅣ가-힣]+', '', string) #대체(한글이 아닌것들->공백)
    string = re.sub('(\s|\t)+', ' ', string) #공백 없애주기
    if _morph:
        m = morph.pos(string, norm=True, stem=True) #pos : (형태소,종류) tuple list 반환, normalize, stemming
        string = ' '.join(['%s/%s'%(p[0],p[1]) for p in #합치면서 앞에 공백한칸씩 사랑/Noun
            filter(lambda x: x[1] in ['Noun', 'Verb', 'Adjective', 'Adverb'],
            # filter(lambda x: x[1] != 'Suffix' and x[1] != 'Josa',
                m
                   #물어보기
            )
        ])
    return string.strip()

def load_data_and_labels(emo_class=[], _morph=False):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    #왜있어??
    return load_data_and_labels2(emo_class, _morph)[:2]

    # # Load data from files
    # emo_texts = dict()
    # emo_index = dict()
    # for i, emo in enumerate(emo_class):
    #     texts = list(open('./corpus/emo/%s' % emo, 'r').readlines())
    #     texts = [s.strip() for s in texts]
    #     emo_texts[emo] = texts
    #     emo_index[emo] = i
    #
    # emo_texts['anger'] = emo_texts['anger'][:1500]
    #
    # emo_labels = {e:[0 for _ in range(len(emo_class))] for e in emo_class}
    #
    # for e in emo_class:
    #     emo_labels[e][emo_index[e]] = 1
    #
    # x_text = []
    # y = []
    # for e in emo_class:
    #     x_text += emo_texts[e]
    #     y += [emo_labels[e] for _ in range(len(emo_texts[e]))]
    # x_text = [clean_str(sent, _morph) for sent in x_text]
    #
    # print('x_text', len(x_text))
    # print('y', len(y))
    #
    # return [x_text, y]

def load_data_and_labels2(emo_class=[], _morph=False):
    """
    2017.02.03 : 원본 text도 return 하도록 수정
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    emo_texts = dict() #dictionary 자료형 생성
    emo_index = dict()
    for i, emo in enumerate(emo_class): #enumerate:인덱스 포함 튜플 반환 i : 인덱스, emo: data
        texts = list(open('./corpus/emo-16.12.07/%s' % emo, 'r',encoding='UTF8').readlines())
        texts = [s.strip() for s in texts] #양끝 공백 떼주기
        emo_texts[emo] = texts
        emo_index[emo] = i

        
        #물어보기
    # emo_texts['anger'] = emo_texts['anger'][:1500]
    emo_labels = {e:[0 for _ in range(len(emo_class))] for e in emo_class} #ONE-hot encoding(가장높은거를 1로) 감정이 해당되면 1 아니면 0 크기는 감정분류수 output나타냄

    for e in emo_class:
        print("# of {} : {}".format(e, len(emo_texts[e])))
        emo_labels[e][emo_index[e]] = 1

    x_text = []
    x_text_orig = []
    y = []
    for e in emo_class:
        x_text += emo_texts[e]
        y += [emo_labels[e] for _ in range(len(emo_texts[e]))]
    x_text_orig = x_text[:]
    x_text = [clean_str(sent, _morph) for sent in x_text]

    print('x_text', len(x_text))
    print('y', len(y))

    return [x_text, x_text_orig, y] #Y:OUTPUT(감정), X_TEXT:형태소 변환후 문장. X_TEXT_OR:변환전문장

def make_doc_with_data_and_labels(texts, labels):
    if type(labels) != list:
        labels = labels.tolist()
    doc = ['' for _ in range(len(labels[0]))]
    for text, label in zip(texts, labels):
        ind = label.index(max(label))
        doc[ind] += text + '\n'

    return doc

def make_voca_and_train_dev(x_text, x_text_orig, y, voca=None):
    # 2017.02.03 : 원본 text도 return 하도록 수정
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text]) #띄어쓰기로 구분
    if voca:
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(voca) #learn data 훈련
        vocab_processor.max_document_length = max_document_length
    else:
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.transform(x_text))) #Transform documents to word-id matrix. Convert words to ids with vocabulary fitted with fit or the one provided in the constructor.
    y = np.array(y)

    x_text = np.array(x_text)
    x_text_orig = np.array(x_text_orig)

    # Randomly shuffle data
    np.random.seed(10) 
    shuffle_indices = np.random.permutation(np.arange(len(y))) #순서를 임의로 바꾸거나 임의의 순열을 반환한다. np.arange - 0 ~ y 까지 배열


    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    x_text_shuffled = x_text[shuffle_indices]
    x_text_orig_shuffled = x_text_orig[shuffle_indices]

    
    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    train_cnt = (int) (x_shuffled.shape[0] * 0.15) #shape : 각 차원의 크기 곱하기 0.15를 test 보통 10~20%
    x_train, x_dev = x_shuffled[:-train_cnt], x_shuffled[-train_cnt:] #85% 트레이닝이다.
    y_train, y_dev = y_shuffled[:-train_cnt], y_shuffled[-train_cnt:] #감정을 섞어서 학습시켜야 능률 올라가
    x_train_text, x_dev_text = x_text_shuffled[:-train_cnt], x_text_shuffled[-train_cnt:]
    x_train_text_orig, x_dev_text_orig = x_text_orig_shuffled[:-train_cnt], x_text_orig_shuffled[-train_cnt:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    return vocab_processor, x_train, y_train, x_dev, y_dev, x_train_text, x_dev_text, x_train_text_orig, x_dev_text_orig


#언제쓰는거야? 데이터셋이 많을경우
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data) #계산위해 배열 편하게 쓰려고  numpy
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__ == "__main__":
    load_data_and_labels(emo_class=['contempt', 'happiness', 'awo', 'sadness'])
