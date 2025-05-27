import numpy as np

import importance_model as im
import preprocessing as pre

def summerizer(text, title, keyWords, embeddingMatrix, precentSummerization=0.3):
    words = pre.clean_and_tockenize(text)
    vocab = im.buildVocab(words)
    sentences = im.sentencesSplitter(text)
    features = im.extract_features(text, embeddingMatrix, vocab, keyWords, title)
    ranking = [featureOfsentence["importanceScore"] for featureOfsentence in features]
    numSentences = int(np.round(len(ranking) * precentSummerization))

    print(f"Ranking: {np.array(ranking)}")

    indexSentences = np.argsort(ranking)[::-1]
    indexSentences = indexSentences[:numSentences+1]

    indexSentences.sort()
    summerizedSentences = [sentences[i] for i in indexSentences]

    for sentence in summerizedSentences:
        print(sentence)





