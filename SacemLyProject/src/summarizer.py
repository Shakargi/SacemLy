import numpy as np
import LSA_keywords as lsa
import importance_model as im
import preprocessing as pre


def summarizer(text, title, embeddingMatrix, numSentences=20):
    words = pre.clean_and_tockenize(text)
    vocab = im.buildVocab(words)
    sentences = im.sentencesSplitter(text)
    keyWords = lsa.keywords(sentences)
    print(keyWords)
    features = im.extract_features(text, embeddingMatrix, vocab, keyWords, title)

    keyWordDensity = [featureOfsentence["keywordDensity"] for featureOfsentence in features]
    similarityToTitle = [featureOfsentence["similarityToTitle"] for featureOfsentence in features]
    importanceScore = [featureOfsentence["importanceScore"] for featureOfsentence in features]

    score = 0.5 * np.array(importanceScore) + 0.3 * np.array(similarityToTitle) + 0.2 * np.array(keyWordDensity)

    if numSentences > len(score):
        numSentences = max(int(np.ceil(len(score) * 0.2)), 1)


    indexSentences = np.argsort(score)[::-1]
    indexSentences = indexSentences[:numSentences]

    indexSentences.sort()
    summerizedSentences = [sentences[i] for i in indexSentences]

    return summerizedSentences


