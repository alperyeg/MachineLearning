from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 100000


def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower().decode('utf8'))
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print len(l2)
    return l2


def sample_handling(sample, lexicon, classification):
    featureset = []
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower().decode('utf8'))
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower().decode('utf8'))
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset


def create_feature_set_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1, 0])
    features += sample_handling('neg.txt', lexicon, [0, 1])
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size * len(features))
    X_train = list(features[:, 0][:-testing_size])
    y_train = list(features[:, 1][:-testing_size])
    X_test = list(features[:, 0][-testing_size:])
    y_test = list(features[:, 1][-testing_size:])

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = create_feature_set_and_labels(
        'pos.txt', 'neg.txt', test_size=0.1)
    print len(X_train), len(y_train), len(X_test), len(y_test)
    print 'Starting to pickle...'
    with open('sentiment_pickle', 'wb') as f:
        pickle.dump([X_train, y_train, X_test, y_test], f)
