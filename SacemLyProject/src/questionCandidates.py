import nltk
import numpy as np

from preprocessing import clean_and_tockenize
from importance_model import sentencesSplitter


nltk.download('words')
def isCandidate(sentence: str):
    words = sentence.split()
    if len(words) < 5:
        return False, []

    if sentence[-1] == '?':
        return False, []

    tokens = clean_and_tockenize(sentence)
    posTags = nltk.pos_tag(tokens)

    tags = []
    hasVerb = False
    hasNoun = False

    for _, tag in posTags:
        tags.append(tag)
        if "NN" in tag:
            hasNoun = True
        if "VB" in tag:
            hasVerb = True

    return hasNoun and hasVerb, tags


def getCandidate(text: str):
    sentences = sentencesSplitter(text)
    goodSentences = []
    for sentence in sentences:
        isCandidateTerm, tags = isCandidate(sentence)
        if isCandidateTerm:
            goodSentences.append((sentence, tags))

    return goodSentences


def rateQuestionCandidates(text: str):
    sentences = getCandidate(text)
    scoring = np.zeros(len(sentences))
    NEGATION_WORDS = [
        "no", "not", "n't", "never", "nothing", "nowhere", "neither", "none", "nobody", "nor",
        "cannot", "can't", "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't",
        "wasn't", "weren't", "isn't", "aren't", "ain't", "haven't", "hasn't", "hadn't", "without"
    ]

    for i in range(len(sentences)):
        tags = sentences[i][1]
        sentence = sentences[i][0].lower()
        words = sentence.split()

        # checks for subordinating conjunctions at the beginning of the sentence
        if tags[0] != "IN":
            scoring[i] += 0.2
        else:
            scoring[i] += 0.1
        # checks if there are negation words
        hasNegationWord = False
        for word in words:
            if word in NEGATION_WORDS:
                hasNegationWord = True

        if not hasNegationWord:
            scoring[i] += 0.2
        else:
            scoring[i] -= 0.1

        # checks if there are named entities
        chunked = nltk.ne_chunk(list(zip(words, tags)))

        for subtree in chunked:
            if isinstance(subtree, nltk.Tree):
                scoring[i] += 0.2

        # checks if there are is syntactic structure (noun before the verb)
        hasStructure = False
        for j in range(len(tags)):
            if "NN" in tags[j]:
                noun = j
                for k in range(j+1, len(tags)):
                    if "VB" in tags[k]:
                        hasStructure = True

        if hasStructure: scoring[i] += 0.2

        # checks the amount of content words in sentence
        amount = 0
        for tag in tags:
            if "NN" in tag or "VB" in tag or "JJ" in tag or "RB" in tag:
                amount+=1

        scoring[i] += (amount / len(words) * 0.2)

    return scoring



