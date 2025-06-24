import preprocessing as pre
import numpy as np
import re
import json
import os
import importance_model as im
from LSA_keywords import keywords


def calcAverageSentence(text: str):
    sentences = im.sentencesSplitter(text)
    lengths = [len(pre.clean_and_tockenize(sentence)) for sentence in sentences]
    return np.mean(lengths)


def countPunctuation(text: str):
    return {".": text.count('.'),
            ",": text.count(','),
            ";": text.count(';'),
            "?": text.count('?'),
            "!": text.count('!'),
            "-": text.count('-'),
            '"': text.count('"')}


def calcAverageStanzas(text: str):
    numSentences = len(im.sentencesSplitter(text))
    paragraphs = len([p.strip() for p in re.split(r'[\n\t]+', text)])

    if paragraphs == 0:
        return 0

    return numSentences / paragraphs

def meanAndStd(vector):
    vector = np.array(vector)
    if len(vector) == 0 or np.all(np.isnan(vector)) or np.std(vector) == 0:
        return float(np.mean(vector)), 0.0
    return float(np.mean(vector)), float(np.std(vector))



def StyleWrite(text, embeddingMatrix, vocab, keywords, title = ""):
    features = im.extract_features(text, embeddingMatrix, vocab, keywords, title)
    keyWordDensities = np.array([features[i]["keywordDensity"] for i in range(len(features))])
    similarityToTitles = np.array([features[i]["similarityToTitle"] for i in range(len(features))])
    importanceScores = np.array([features[i]["importanceScore"] for i in range(len(features))])

    keyWordDensitiesMean, keyWordDensitiesSTD = meanAndStd(keyWordDensities)
    similarityToTitlesMean, similarityToTitlesSTD = meanAndStd(similarityToTitles)
    importanceScoresMean, importanceScoresSTD = meanAndStd(importanceScores)


    return {"avg_sentence_length": calcAverageSentence(text),
            "punctuation_freq": countPunctuation(text),
            "avg_sentences_per_paragraph": calcAverageStanzas(text),
            "keyWordDensitiesMean" : keyWordDensitiesMean,
            "keyWordDensitiesSTD" : keyWordDensitiesSTD,
            "similarityToTitlesMean" : similarityToTitlesMean,
            "similarityToTitlesSTD" : similarityToTitlesSTD,
            "importanceScoresMean" : importanceScoresMean,
            "importanceScoresSTD" : importanceScoresSTD}

def StyleVector(text, embeddingMatrix, vocab, Keywords, title=""):

    properties = StyleWrite(text, embeddingMatrix, vocab, Keywords, title)
    vector = [properties["avg_sentence_length"]]
    punctuation_order = [".", ",", ";", "?", "!", "-", '"']
    for p in punctuation_order:
        vector.append(properties["punctuation_freq"].get(p, 0))

    vector.append(properties["avg_sentences_per_paragraph"])
    vector.append(properties["keyWordDensitiesMean"])
    vector.append(properties["keyWordDensitiesSTD"])
    vector.append(properties["similarityToTitlesMean"])
    vector.append(properties["similarityToTitlesSTD"])
    vector.append(properties["importanceScoresMean"])
    vector.append(properties["importanceScoresSTD"])

    return np.array(vector)


def compareStyles(text1, text2, embeddingMatrix, vocab, Keywords, title=""):
    v1 = StyleVector(text1, embeddingMatrix, vocab, keywords, title)
    v2 = StyleVector(text2, embeddingMatrix, vocab, keywords, title)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 != 0 and norm2 != 0:
        return np.dot(v1, v2) / (norm1 * norm2)

    return 0

def loadStyleProfile():
    with open("styleProfile.json", "r") as file:
        profile = json.load(file)
    return profile

def updateStyleProfile(newStyleVector, learningRate=0.2, filename="styleProfile.json"):
    newStyleVector = newStyleVector.tolist()  # המרת numpy array ל-list
    if os.path.exists(filename):
        with open(filename, "r") as file:
            try:
                profile = json.load(file)
            except json.JSONDecodeError:
                profile = None
    else:
        profile = None

    if profile is None or "style_vector" not in profile:
        profile = {"style_vector": newStyleVector, "samples": 1}
    else:
        old_vector = np.array(profile["style_vector"])
        updated_vector = (1 - learningRate) * old_vector + learningRate * np.array(newStyleVector)
        profile["style_vector"] = updated_vector.tolist()
        profile["samples"] = profile.get("samples", 1) + 1

    with open(filename, "w") as file:
        json.dump(profile, file, indent=4)


def comparingCurrentStyle(text,  embeddingMatrix, vocab, keywords, title=""):
    v1 = StyleVector(text, embeddingMatrix, vocab, keywords, title)
    v2 = np.array(loadStyleProfile()["style_vector"])

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 != 0 and norm2 != 0:
        return np.dot(v1, v2) / (norm1 * norm2)

    return 0

def cleanStats(filename="styleProfile.json"):
    profile = {"style_vector": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "samples": 0}
    with open(filename, "w") as file:
        json.dump(profile, file, indent=4)



