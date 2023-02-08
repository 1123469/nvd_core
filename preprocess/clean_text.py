import re
from nltk.stem import WordNetLemmatizer


def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    lemmatizer = WordNetLemmatizer()
    lem_words = [lemmatizer.lemmatize(w, pos='n') for w in words]
    stopwords = {}.fromkeys([line.rstrip() for line in open('F:\\PycharmProjects\\NVDproject\\nvdcve\\stopwords.txt')])
    eng_stopwords = set(stopwords)
    words = [w for w in lem_words if w not in eng_stopwords]
    return words