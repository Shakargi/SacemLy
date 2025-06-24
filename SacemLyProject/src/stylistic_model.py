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


def StyleWrite(text, embeddingMatrix, vocab, keywords, title=""):
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
            "keyWordDensitiesMean": keyWordDensitiesMean,
            "keyWordDensitiesSTD": keyWordDensitiesSTD,
            "similarityToTitlesMean": similarityToTitlesMean,
            "similarityToTitlesSTD": similarityToTitlesSTD,
            "importanceScoresMean": importanceScoresMean,
            "importanceScoresSTD": importanceScoresSTD}


def StyleVectorText(text, embeddingMatrix, vocab, Keywords, title=""):
    properties = StyleWrite(text, embeddingMatrix, vocab, Keywords, title)
    vector = [properties["avg_sentence_length"]]
    punctuation_order = [".", ",", ";", "?", "!", "-", '"']
    for p in punctuation_order:
        vector.append(properties["punctuation_freq"].get(p, 0))

    vector.append(properties["avg_sentences_per_paragraph"])


    return np.array(vector)


def StyleVectorSentence(text, embeddingMatrix, vocab, Keywords, title=""):
    properties = StyleWrite(text, embeddingMatrix, vocab, Keywords, title)
    vector = []

    vector.append(properties["keyWordDensitiesMean"])
    vector.append(properties["keyWordDensitiesSTD"])
    vector.append(properties["similarityToTitlesMean"])
    vector.append(properties["similarityToTitlesSTD"])
    vector.append(properties["importanceScoresMean"])
    vector.append(properties["importanceScoresSTD"])

    return np.array(vector)


def compareTextStyles(text1, text2, embeddingMatrix, vocab, keywords, title=""):
    v1_text = StyleVectorText(text1, embeddingMatrix, vocab, keywords, title)
    v2_text = StyleVectorText(text2, embeddingMatrix, vocab, keywords, title)

    v1_sent = StyleVectorSentence(text1, embeddingMatrix, vocab, keywords, title)
    v2_sent = StyleVectorSentence(text2, embeddingMatrix, vocab, keywords, title)

    def cosine_sim(v1, v2):
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        return np.dot(v1, v2) / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0

    sim_text = cosine_sim(v1_text, v2_text)
    sim_sentence = cosine_sim(v1_sent, v2_sent)

    return {
        "TextStyleSimilarity": sim_text,
        "SentenceStyleSimilarity": sim_sentence
    }


def loadStyleProfile():
    with open("styleProfile.json", "r") as file:
        profile = json.load(file)
    return profile


def updateStyleProfile(text_vector, sentence_vector, num_sentences, filename="styleProfile.json"):
    text_vector = np.array(text_vector)
    sentence_vector = np.array(sentence_vector)

    if os.path.exists(filename):
        with open(filename, "r") as file:
            try:
                profile = json.load(file)
            except json.JSONDecodeError:
                profile = None
    else:
        profile = None

    if profile is None or "TextStyleVector" not in profile or "SentenceStyleVector" not in profile:
        profile = {
            "TextStyleVector": text_vector.tolist(),
            "SentenceStyleVector": sentence_vector.tolist(),
            "texts": 1,
            "sentences": num_sentences
        }
    else:
        old_text_vector = np.array(profile["TextStyleVector"])
        old_sentence_vector = np.array(profile["SentenceStyleVector"])
        old_num_texts = profile["texts"]
        old_num_sentences = profile["sentences"]

        updated_text_vector = (old_text_vector * old_num_texts + text_vector) / (old_num_texts + 1)
        updated_sentence_vector = (old_sentence_vector * old_num_sentences + sentence_vector * num_sentences) / (
                    old_num_sentences + num_sentences)

        profile["TextStyleVector"] = updated_text_vector.tolist()
        profile["SentenceStyleVector"] = updated_sentence_vector.tolist()
        profile["texts"] = old_num_texts + 1
        profile["sentences"] = old_num_sentences + num_sentences

    with open(filename, "w") as file:
        json.dump(profile, file, indent=4)


def comparingCurrentStyle(text, embeddingMatrix, vocab, keywords, title=""):
    text_vector = StyleVectorText(text, embeddingMatrix, vocab, keywords, title)
    sentence_vector = StyleVectorSentence(text, embeddingMatrix, vocab, keywords, title)

    profile = loadStyleProfile()
    profile_text_vector = np.array(profile["TextStyleVector"])
    profile_sentence_vector = np.array(profile["SentenceStyleVector"])

    def cosine_sim(v1, v2):
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        return np.dot(v1, v2) / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0

    sim_text = cosine_sim(text_vector, profile_text_vector)
    sim_sentence = cosine_sim(sentence_vector, profile_sentence_vector)

    return {
        "TextStyleSimilarity": sim_text,
        "SentenceStyleSimilarity": sim_sentence
    }


def cleanStats(filename="styleProfile.json"):
    profile = {
        "TextStyleVector": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "SentenceStyleVector": [0, 0, 0, 0, 0, 0],
        "texts": 0,
        "sentences": 0
    }
    with open(filename, "w") as file:
        json.dump(profile, file, indent=4)





