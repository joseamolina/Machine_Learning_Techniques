from math import log

class Model:

    def __init__(self, name_positive, name_negative):

        self.negative = {}
        self.neg_P = {}

        self.positive = {}
        self.pos_P = {}

        self.set_words = set()

        self.reader_neg = open('dataFiles/train/trainNeg.txt', 'r', encoding="ISO-8859-1")
        self.reader_pos = open('dataFiles/train/trainPos.txt', 'r', encoding="ISO-8859-1")

        self.count_dic_neg = 0
        self.count_dic_pos = 0

        self.V = 0

        self._train()

    def _train(self):

        for i in self.reader_neg:
            lista = i.split()
            for x in lista:
                if x[0] != '@':
                    self.set_words.add(x)
                    value = self.negative.get(x)
                    if value == None:
                        self.negative[x] = 1
                        self.neg_P[x] = None
                    else:
                        self.negative[x] = value + 1

        for i in self.reader_pos:
            lista = i.split()
            for x in lista:
                if x[0] != '@':
                    self.set_words.add(x)
                    value = self.positive.get(x)
                    if value == None:
                        self.positive[x] = 1
                        self.pos_P[x] = None
                    else:
                        self.positive[x] = value + 1

        self.V = len(self.set_words)

        self.count_dic_neg = len(self.negative)
        self.count_dic_pos = len(self.positive)

        #prob_neg = count_dic_neg / count_dic_neg + count_dic_pos
        for i in self.negative:
            self.neg_P[i] = log((self.negative[i] + 1)/(self.count_dic_neg + self.V))

        #prob_pos = count_dic_pos / count_dic_neg + count_dic_pos
        for i in self.positive:
            self.pos_P[i] = log((self.positive[i] + 1)/(self.count_dic_pos + self.V))

    def _chech_tweet(self, tweet):
        tweet = tweet.split()

        summer_pos = log(len(self.positive) / len(self.positive) + len(self.negative))
        summer_neg = log(len(self.negative) / len(self.positive) + len(self.negative))

        for x in range(1, len(tweet)):

            if not self.pos_P.get(tweet[x]):
                var = log(1 / self.count_dic_pos + self.V)
            else:
                var = self.pos_P[tweet[x]]
            summer_pos += var

            if not self.neg_P.get(tweet[x]):
                var = log(1 / self.count_dic_neg + self.V)
            else:
                var = self.neg_P[tweet[x]]
            summer_neg += var

        return 1 if summer_pos > summer_neg else -1

    def check_accuracy(self, testPos, testNeg):

        reader_neg_test = open(testNeg, 'r', encoding="ISO-8859-1")
        reader_pos_test = open(testPos, 'r', encoding="ISO-8859-1")

        tester_neg = reader_neg_test.read()
        lista_neg = tester_neg.split("@")
        list_tweets_neg = list(map(lambda k: k.replace("\n", "").replace("\"", "").replace(".", ""), lista_neg))

        tester_pos = reader_pos_test.read()
        lista_pos = tester_pos.split("@")
        list_tweets_pos = list(map(lambda k: k.replace("\n", "").replace("\"", "").replace(".", ""), lista_pos))

        n_pos = 0
        n_neg = 0
        for i in list_tweets_neg:
            rt = self._chech_tweet(i)
            if rt < 0:
                n_neg += 1
            else:
                n_pos += 1

        print('The accuracy of test is {0}% of correct predictions of negative dataset'.format((100 * n_neg) / (n_neg + n_pos)))

        n_pos = 0
        n_neg = 0
        for i in list_tweets_pos:
            rt = self._chech_tweet(i)
            if rt < 0:
                n_neg += 1
            else:
                n_pos += 1


        print(len(self.set_words))
        print('The accuracy of test is {0}% of correct predictions of positive dataset'.format((100 * n_pos) / (n_neg + n_pos)))