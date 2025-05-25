import re
import nltk
import numpy as np
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('stopwords')
stopWords = stopwords.words('english')

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



