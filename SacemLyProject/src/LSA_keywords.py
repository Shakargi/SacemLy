import numpy as np
import sklearn.feature_extraction.text as text

def buildTFIDFMatrix(documents):
    if not any(doc.strip() for doc in documents):
        return None, None
    vectorizer = text.TfidfVectorizer(stop_words='english')
    try:
        tfIdfMatrix = vectorizer.fit_transform(documents)
        return vectorizer, tfIdfMatrix
    except ValueError:
        return None, None


def keywords(documents, wordsPerTopic = 2, numTopics = 4):
    keyWords = set()
    vectorizer, tfIdfMatrix = buildTFIDFMatrix(documents)

    if tfIdfMatrix is None:
        return []

    tfIdfMatrix = tfIdfMatrix.toarray()

    if tfIdfMatrix.shape[0] == 0 or tfIdfMatrix.shape[1] == 0:
        return []

    U, sigma, Vt = np.linalg.svd(tfIdfMatrix)


    vocab = vectorizer.get_feature_names_out()
    numTopics = min(numTopics, Vt.shape[0])

    for i in range(numTopics):
        sortedIndices = np.argsort(Vt[i])[::-1]
        topWordIndices = sortedIndices[:min(wordsPerTopic, len(sortedIndices))]
        for j in range(len(topWordIndices)):
            keyWords.add(vocab[topWordIndices[j]])

    return list(keyWords)




