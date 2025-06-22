import preprocessing as pre
import importance_model as im
import numpy as np
import re


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


def StyleWrite(text):
    return {"avg_sentence_length": calcAverageSentence(text),
            "punctuation_freq": countPunctuation(text),
            "avg_sentences_per_paragraph": calcAverageStanzas(text)}
