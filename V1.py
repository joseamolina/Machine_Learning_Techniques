
def voc_comp():

    neg_dic = {}

    pos_dic = {}



    reader_neg = open('dataFiles/train/trainNeg.txt', 'r', encoding="ISO-8859-1")
    reader_pos = open('dataFiles/train/trainPos.txt', 'r', encoding="ISO-8859-1")


    pos_neg = reader_neg.read().split("@")

    for tweet in pos_neg:
        tweet = tweet.split()
        tweet.pop(0)

        for r in tweet:

            if neg_dic.get(r) != None:
                neg_dic[r] = 1

            else:
                neg_dic[r] = neg_dic[r] + 1

        print(tweet)

    pos_pos = reader_pos.read().split("@")

    for tweet in pos_pos:
        tweet = tweet.split()
        tweet.pop(0)
        for r in tweet:

            if neg_dic.get(r) != None:
                pos_dic[r] = 1

            else:
                pos_dic[r] = neg_dic[r] + 1

def passer():
    vocab = set()


    posContents = posFileObj.read()
    negContents = negFileObj.read()
    # num_pos_tweets = len(posContents)/(len(posContents)+len(negContents))
    # num_neg_tweets = len(negContents)/(len(posContents)+len(negContents))
    posWords = posContents.split()
    negWords = negContents.split()
    #print(negWords)
    #print(num_neg_tweets)
    #print(num_pos_tweets)

    vocab.update(posWords)
    vocab.update(negWords)
    #print(len(vocab))

    posDict=dict.fromkeys(posWords, 0)
    negDict=dict.fromkeys(negWords, 0)
    #print(posDict)

    for w in posWords:
        posDict[w]+=1

    for word in posDict:
        posDict[word] = (posDict[word]+1)/(len(vocab)+2)


    for w in negWords:
        negDict[w]+= 1

    for word in negDict:
        negDict[word] = (negDict[word]+1)/(len(vocab)+2)
        #print(negDict[word])

    #print(negDict['bad'])

    posFileObj1 = codecs.open("/home/local/STUDENT-CIT/r00171285/Desktop/dataFiles/test/testPos.txt", 'r', encoding="ISO-8859-1")
    negFileObj1 = codecs.open("/home/local/STUDENT-CIT/r00171285/Desktop/dataFiles/test/testNeg.txt", 'r', encoding="ISO-8859-1")

    posTweets = posFileObj1.read()
    posContent1 = posTweets.splitlines()
    negTweets = negFileObj1.read()
    negContent1 = negTweets.splitlines()
    #print(negContent1)
    # testTweets = str()
    # testTweets = posContent1
    # testTweets.extend(negContent1)
    #print(len(testTweets))

    total_pos_sen = 0
    total_neg_sen = 0

    for tweets in posContent1:
        posCount = 0
        negCount = 0
        words_list = tweets.split()
        for word in words_list:
            if word in posDict:
                posCount += posDict[word]
            if word in negDict:
                negCount += negDict[word]
        if posCount > negCount:
            total_pos_sen += 1
        else:
            total_neg_sen += 1

    print(total_pos_sen)
    print(total_neg_sen)

        # for i in words_list:
        #     if i in vocab:
        #         posProb[tweets] += posDict[i]
        #         negProb[tweets] += negDict[i]
        #     else:
        #         posProb[tweets] += 0
        #         negProb[tweets] += 0
        # print(posProb, negProb)
            # if posProb[tweets]>negProb[tweets]:
            #     print("\n"+ tweets + " is positive")
            # else:
            #     print('\n'+ tweets +' is Negative')

voc_comp()