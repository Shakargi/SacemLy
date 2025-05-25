import numpy as np
from cbow_model import fit
from preprocessing import clean_and_tockenize
import re

def buildVocab(words):
    uniqueWords = sorted(set(words))
    return {Word : i for i, Word in enumerate(uniqueWords)}

def buildReverseVocab(words):
    uniqueWords = sorted(set(words))
    return {i: Word for i, Word in enumerate(uniqueWords)}




def generatePairs(words, windowSize):
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

def sentencesSplitter(text):
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    return sentences


def Sentences2Vectors(sentences, vocab, embeddingMatrix):
    vectors = []
    for sentence in sentences:
        words = clean_and_tockenize(sentence)

        filtered_words = [w for w in words if w in vocab]
        if not filtered_words:
            continue

        word_vectors = [embeddingMatrix[vocab[word]] for word in filtered_words]
        sentenceVector = np.mean(word_vectors, axis=0)
        vectors.append(sentenceVector)


    return vectors



def compute_position_score(index, totalSentences):
    return index / totalSentences

def compute_length_score(sentenceLength, maxLength):
    return sentenceLength / maxLength


def compute_keyword_density(sentenceTokens, keywords):
    tokensInKeywords = 0
    for token in sentenceTokens:
        if token in keywords:
            tokensInKeywords += 1

    return tokensInKeywords / len(sentenceTokens) * 100


def compute_similarity_to_title(sentenceVector, titleVector):
    norm1 = np.linalg.norm(sentenceVector)
    norm2 = np.linalg.norm(titleVector)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(sentenceVector, titleVector) / (norm1 * norm2)


def extract_features(text, embeddingMatrix, vocab, keywords, title=""):
    titleVector = Sentences2Vectors([title], vocab, embeddingMatrix)
    if len(titleVector) == 0:
        titleVector = np.zeros(embeddingMatrix.shape[1])  # וקטור אפס אם הכותרת ריקה או לא נמצאה
    else:
        titleVector = titleVector[0]


    sentences = sentencesSplitter(text)
    vectors = Sentences2Vectors(sentences, vocab, embeddingMatrix)
    tokensPerSentences = [clean_and_tockenize(sentence) for sentence in sentences]
    maxLength = np.max([len(tokensPerSentences[i]) for i in range(len(sentences))])

    positions = [compute_position_score(i, len(sentences)) for i in range(len(sentences))]
    lengths = [compute_length_score(len(tokensPerSentences[i]), maxLength) for i in range(len(sentences))]
    keywordsPrecent = [compute_keyword_density(tokensPerSentences[i], keywords) for i in range(len(sentences))]
    similarities = [compute_similarity_to_title(vectors[i], titleVector) for i in range(len(sentences))]
    featuresPerSentenceDS = []
    for i in range(len(sentences)):
        sentenceFeatures = {"text" : sentences[i],
                            "vector" : vectors[i],
                            "positionScore" : positions[i],
                            "lengthScore" : lengths[i],
                            "keywordDensity" : keywordsPrecent[i],
                            "similarityToTitle" : similarities[i]}


        featuresPerSentenceDS.append(sentenceFeatures)

    return featuresPerSentenceDS

