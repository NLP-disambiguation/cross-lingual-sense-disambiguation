from util import untag_tokens, lemmatize_query, tokenize, translate_si_en, wordnet_similarity
from collections import Counter
import itertools
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

model_vocabulary_en = None


def word2vec(token_list, save=False):
    """
    Make a word2vec model from tokens
    :param token_list: list of tagged tokens
    :return: generated model
    """
    global model_vocabulary_en
    model_vocabulary_en = None
    corpus = untag_tokens(token_list)
    word_emb_model = Word2Vec(sentences=corpus, window=5, min_count=1, workers=4)
    print("vocabulary:", word_emb_model.wv.index_to_key)  # make sure this is words, not characters
    if save: word_emb_model.save("word2vec.model")

    return word_emb_model


def translate(word, model_vocabulary):
    # for words not in model, use closest that is by a measure of sense similarity in English where data is
    # more available through translation

    global model_vocabulary_en
    if model_vocabulary_en == None:
        model_vocabulary_en = []
        for wv in model_vocabulary:
            try:
                t, conf = translate_si_en(wv)
                model_vocabulary_en.append(t)
            except Exception as e: print(wv, e)

    max = ["", -1]
    t, conf = translate_si_en(word)
    for i, wv in enumerate(model_vocabulary_en):
        t = t.replace("to ", "")
        wv = wv.replace("to ", "")
        # print(t, wv)
        s = wordnet_similarity(t, wv)
        # print(s)
        if s > max[1]: max = [model_vocabulary[i], s]
    # print(max)
    return max


def get_sif_feature_vectors(sentence1, sentence2, word_emb_model):
    contained_sentence1 = [token for token in sentence1.split() if token in word_emb_model.wv.index_to_key]
    not_contained_sentence1 = [translate(token, word_emb_model.wv.index_to_key) for token in sentence1.split() if
                               token not in contained_sentence1]
    contained_sentence2 = [token for token in sentence2.split() if token in word_emb_model.wv.index_to_key]
    not_contained_sentence2 = [translate(token, word_emb_model.wv.index_to_key) for token in sentence2.split() if
                               token not in contained_sentence2]
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
        print(c)
        c_tokens = tokenize(text_classes[c])
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


def load_keyed_vector_model(path):
    model = KeyedVectors.load_word2vec_format(path)
    return model


def generate_combinations(one_token_list, token_substitutions, ambiguous_word, current_token_i, sentences):
    """
    Recursively generate token combinations - add a new token to the end of each current sentence
    :param one_token_list: list of tokens for one sentence
    :param token_substitutions: dictionary of similar words for a token
    :param ambiguous_word:
    :param current_token_i:
    :param sentences:
    :return:
    """
    if current_token_i == len(one_token_list):
        return sentences
    new_sentences = []
    if sentences == [] and one_token_list[current_token_i][0] != ambiguous_word:
        for substitution in token_substitutions[one_token_list[current_token_i][0]]:
            new_sentences.append([substitution])
    if sentences == [] and one_token_list[current_token_i][0] == ambiguous_word:
        new_sentences.append([ambiguous_word])
    else:
        if one_token_list[current_token_i][0] == ambiguous_word:
            for sentence in sentences:
                new_sentences.append(sentence + [ambiguous_word])
        else:
            for substitution in token_substitutions[one_token_list[current_token_i][0]]:
                for sentence in sentences:
                    new_sentences.append(sentence + [substitution])
    return generate_combinations(one_token_list, token_substitutions, ambiguous_word, current_token_i + 1,
                                 new_sentences)


def generate_similar(one_token_list, ambiguous_word, model, n_similar=10):
    """
    Generates all possible token combinations of similar words based on input token list.
    :param one_token_list: list of tokens for one sentence
    :param ambiguous_word:
    :param model:
    :return:
    """
    token_substitutions = {}
    for t in one_token_list:
        lemma, upos, data = t
        if lemma != ambiguous_word:
            try:
                generated = model.most_similar_cosmul(positive=[lemma], topn=n_similar)
                generated = tokenize([w[0].replace("_", " ") for w in generated])
                output = []
                for ts in generated:
                    for tok in ts: output += [tok]
                output = [s[0] for s in output if s[1] == upos]
                output.append(lemma)
                token_substitutions[lemma] = list(set(output))
            except Exception as e:
                print("generate_similar", e)
                token_substitutions[lemma] = [lemma]
        else: token_substitutions[lemma] = [lemma]
    all_tokens = []
    for k in token_substitutions.keys():
        all_tokens += token_substitutions[k]
    return all_tokens


def generate_additional_data(data, fname, model, n_similar=10, target_column="meaning"):
    """
    Generate similar sequences of tokens based on input
    :param data: dataframe containing original data
    :param fname: file where the new sequences will be written to
    :param model: word2vec dictionary
    :param n_similar: number of similar words to attempt to generate
    :param target_column: the meaning column name in the dataframe
    :return:
    """
    token_list = tokenize(list(data.sentence.values))
    with open(fname, "w", encoding='utf-8') as file:
        file.write(",".join(data.columns) + "\n")
        for ambigous_word in data.word.unique():
            print(ambigous_word)
            subdata = data[data.word == ambigous_word]
            for i in subdata.index:  # index will be the same as original df
                one_token_list = token_list[i]
                file.write(" ".join([w[0] for w in one_token_list]) + "," + data.iloc[i][target_column] + "," + ambigous_word + "\n")
                similars = generate_similar(one_token_list, ambigous_word, model, n_similar=n_similar)
                file.write(" ".join(similars) + "," + data.iloc[i][target_column] + "," + ambigous_word + "\n")
