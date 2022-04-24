from util import untag_tokens, lemmatize_query, tokenize, translate_si_en, wordnet_similarity
from collections import Counter
import itertools
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

model_vocabulary_en = None

def word2vec(token_list, save=False):
    """
    Make a word2vec model from tokens
    :param token_list: list of tagged tokens
    :return: generated model
    """
    corpus = untag_tokens(token_list)

    word_emb_model = Word2Vec(sentences=corpus, window=5, min_count=1, workers=4)
    if save: word_emb_model.save("word2vec.model")
    print("vocabulary:", word_emb_model.wv.index_to_key)  # make sure this is words, not characters

    return word_emb_model

def translate(word, model_vocabulary):
    global model_vocabulary_en
    if model_vocabulary_en == None:
        model_vocabulary_en = []
        for wv in model_vocabulary:
            t, conf = translate_si_en(wv)
            model_vocabulary_en.append(t)

    max = ["", -1]
    t, conf = translate_si_en(word)
    for i, wv in enumerate(model_vocabulary_en):
        t = t.replace("to ", "")
        wv = wv.replace("to ", "")
        #print(t, wv)
        s = wordnet_similarity(t, wv)
        #print(s)
        if s > max[1]: max = [model_vocabulary[i], s]
    # print(max)
    return max

def get_sif_feature_vectors(sentence1, sentence2, word_emb_model):
    contained_sentence1 = [token for token in sentence1.split() if token in word_emb_model.wv.index_to_key]
    not_contained_sentence1 = [translate(token, word_emb_model.wv.index_to_key) for token in sentence1.split() if token not in contained_sentence1]
    contained_sentence2 = [token for token in sentence2.split() if token in word_emb_model.wv.index_to_key]
    not_contained_sentence2 = [translate(token, word_emb_model.wv.index_to_key) for token in sentence2.split() if token not in contained_sentence2]
    not_contained_sentence1 = [t[0] for t in not_contained_sentence1 if t[1] > 0.5]
    not_contained_sentence2 = [t[0] for t in not_contained_sentence2 if t[1] > 0.5]
    sentence1 = contained_sentence1 + not_contained_sentence1
    sentence2 = contained_sentence2 + not_contained_sentence2
    word_counts = (sentence1 + sentence2)
    word_counts = Counter(itertools.chain(*word_counts))
    embedding_size = 100  # size of vectors in word embeddings
    a = 0.001
    sentence_set = []
    for sentence in [sentence1, sentence2]:
        vs = np.zeros(embedding_size)
        sentence_length = len(sentence)
        for word in sentence:
            a_value = a / (a + word_counts[word])  # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, word_emb_model.wv[word]))  # vs += sif * word_vector
        vs = np.divide(vs, sentence_length)  # weighted average
        sentence_set.append(vs)
    return sentence_set


def Disambiguation_w2v(sentence, text_classes, word_emb_model):
    sentence = lemmatize_query(sentence)[0]
    similarities = {}
    for c in text_classes.keys():
        c_tokens = tokenize(" ".join(text_classes[c]))
        c_sentences = [" ".join(ws) for ws in untag_tokens(c_tokens)]
        sim = 0
        for s in c_sentences:
            v1, v2 = get_sif_feature_vectors(s, sentence, word_emb_model)
            try:
                sim += (cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]) * (1 / len(c_sentences))
            except:
                pass
        similarities[c] = sim
    return similarities
