from util import untag_tokens, lemmatize_query
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf(token_list):
    """
    Make tfidf vectorizer and get top 10 uni/bigrams
    :param token_list: list of tagged tokens
    :return: vectorizer, vectors, dataframe of top n grams
    """
    corpus = [" ".join(ws) for ws in untag_tokens(token_list)]
    # creating vocabulary using uni-gram and bi-gram
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    feature_vectors = tfidf_vectorizer.fit_transform(corpus)

    df = pd.DataFrame(feature_vectors[0].T.todense(), index=tfidf_vectorizer.get_feature_names_out(),
                      columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    top_ngrams = df.head(10)

    return tfidf_vectorizer, feature_vectors, top_ngrams


def Disambiguation_tfidf(query, tfidf_vectorizer, feature_vectors):
    """
    Return the closest matches to the corpus
    :param query: string sencence
    :param tfidf_vectorizer:
    :param feature_vectors:
    :return: calculated similarities, index of closest match that is the same as index of corpus data entry
    """
    query = lemmatize_query(query)
    query_tfidf = tfidf_vectorizer.transform(query)
    cosineSimilarities = cosine_similarity(query_tfidf, feature_vectors).flatten()

    return cosineSimilarities, np.argmax(cosineSimilarities)
