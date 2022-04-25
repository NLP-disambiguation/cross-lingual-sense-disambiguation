import sklearn
import nltk
from nltk.corpus import stopwords
import pandas as pd
import classla
import numpy as np
from nltk.corpus import wordnet as wn
from googletrans import Translator  # pip3 install googletrans==3.1.0a0
import requests, zipfile
from io import BytesIO
import os
import shutil
import gzip

# example for determining meanings and english translation: https://babelnet.org/search?word=zebra&lang=SL&transLang=EN


classla.download('sl')  # download standard models for Slovenian
nlp = classla.Pipeline('sl', processors='tokenize,pos,lemma,depparse')

slo_stopwords = stopwords.words('slovene')
nltk.download('punkt')
nltk.download('stopwords')


def load(path, transform=False, transform_column=None):
    """
    Loads csv file in format <sentence, meaning>
    :param path: path to csv file
    :param transform: should class labels be converted to numeric
    :param transform_column: which column contains labels to be converted
    :return: parsed dataframe and dictionary of sentences for each class
    """
    data = pd.read_csv(path)
    if transform:  # Transform class values to 1/0s
        label_encoder = sklearn.preprocessing.LabelEncoder()
        data[transform_column] = label_encoder.fit_transform(data[transform_column])
    text_classes = {}
    for id in data.meaning.unique():
        text_classes[id] = data[data.meaning == id].sentence.values
    return data, text_classes


def tokenize(text):
    """
    Tokenize given text in Slovene
    :param text: string
    :return: list of tagged tokens
    """
    doc = nlp(text)
    # print(doc.to_conll())
    token_list = []
    for sentence in doc.sentences:
        tokens = [t.to_dict()[0] for t in sentence.tokens]
        tokens = [[t["lemma"], t["upos"], t["feats"] if "feats" in t.keys() else None]
                  for t in tokens
                  if t["lemma"] not in slo_stopwords and t["upos"] != "PUNCT"]
        token_list.append(tokens)
    return token_list


def untag_tokens(token_list):
    """
    Transform list of tagged tokens to list of tokens
    :param token_list: tagged token list
    :return: token list
    """
    corpus = []
    for collection in token_list:
        w = [i[0] for i in collection]
        corpus.append(list(set(w)))
    return corpus


def lemmatize_query(query):
    query = untag_tokens(tokenize(query))
    return [" ".join(ws) for ws in query]


def wordnet_similarity(w1, w2):
    """
    Calculate sense similarity based on Leacock Chodorow method:
        Return a score denoting how similar two word senses are, based on the
        shortest path that connects the senses (as above) and the maximum depth
        of the taxonomy in which the senses occur. The relationship is given as
        -log(p/2d) where p is the shortest path length and d is the taxonomy
        depth.
    :param w1: string
    :param w2: string
    :return: similarity value greater than 0
    """
    w1 = wn.synsets(w1)
    w2 = wn.synsets(w2)
    similarity = 0
    for sense1 in w1:
        for sense2 in w2:
            try:
                s = sense1.lch_similarity(sense2)
                if s > similarity: similarity = s
            except:
                pass
    return similarity


def translate_si_en(word, reverse=False):
    """
    Translate the given slovenian word into english.
    Note: if using sentences, it will not be a word for word translation
    :param word: string
    :return: translated word and confidence
    """
    translator = Translator()
    if not reverse:
        translation = translator.translate(word, src="sl", dest="en")
    else:
        translation = translator.translate(word, src="en", dest="si")
    confidence = translation.extra_data['confidence']
    if translation.text != word:
        translation = translation.text
    else:
        translation = translation.extra_data['all-translations'][0][1][0]

    return translation, confidence


def download_unzip(url):
    print('Downloading started')
    req = requests.get(url)
    print('Downloading Completed')
    # extracting the zip file contents
    zippedfile = zipfile.ZipFile(BytesIO(req.content))
    zippedfile.extractall()


def download_gzip(url):
    filename = url.split('/')[-1]
    req = requests.get(url, stream=True)
    with open(filename, 'wb') as location:
        shutil.copyfileobj(req.raw, location)
    with gzip.open(filename, 'rb') as f_in:
        with open("extracted_" + filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return "extracted_" + filename


def load_downloaded_model(file_path):
    # in form word w1 w2 ... wn
    word2vec = dict()
    i = 0
    with open(file_path, "r") as file:
        line = file.readline()
        while len(line) > 0:
            if i % 100000 == 0: print(i)
            i += 1
            try:
                key = line.split(' ', 1)[0]  # the first word is the key
                value = np.array([float(val) for val in line.split(' ')[1:]], dtype=np.float32)
                word2vec[key] = value
            except:
                print(line)
                break
            line = file.readline()
    return word2vec
