import numpy as np


def buildVocab(words):
    uniqueWords = sorted(set(words))
    return {Word : i for i, Word in enumerate(uniqueWords)}

def buildReverseVocab(words):
    uniqueWords = sorted(set(words))
    return {i: Word for i, Word in enumerate(uniqueWords)}




def generatePairs(words, windowSize = 2):
    data = []
    for i in range(windowSize, len(words) - windowSize):
        context = []
        for j in range(-windowSize, windowSize+1):
            if j!=0:
                context.append(words[i+j])
        data.append((context, words[i]))


    return data


def oneHotEncoder(vocab, word):
    index = vocab[word]
    return np.eye(len(vocab))[index]

def onHotDecoder(reverseVocab, vector):
    return reverseVocab[np.argmax(vector)]


def prepareData(vocab, pairs):
    data = []
    for i in range(len(pairs)):
        onehot = np.zeros(len(vocab))
        context = pairs[i][0]
        target = pairs[i][1]
        for j in range(len(context)):
            onehot += oneHotEncoder(vocab, context[j])
        onehot /= len(context)
        data.append((onehot, oneHotEncoder(vocab, target)))

    return data
