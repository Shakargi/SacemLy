import re
import numpy as np



def clean_and_tockenize(text):

    cleanedText = text.lower()
    cleanedText = re.sub(r'[^\w\s]', '', cleanedText).split()

    # lemmatizer = nltk.stem.WordNetLemmatizer()
    # cleanedFinal = []
    # for word in cleanedText:
    #     if not word in stopWords:
    #         wordAdd = lemmatizer.lemmatize(word)
    #         cleanedFinal.append(wordAdd)
    #
    # return np.array(cleanedFinal)

    return np.array(cleanedText)
