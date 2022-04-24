from util import load, tokenize
from tfidf import tfidf, Disambiguation_tfidf
from word2vec import word2vec, Disambiguation_w2v


if __name__ == "__main__":
    data, text_classes = load('data.csv')
    corpus = " ".join(list(data.sentence.values))
    token_list = tokenize(corpus)

    # relies a corpus matches, faster
    tfidfvectorizer, tfidfvectors, top_ngrams = tfidf(token_list)
    matches, biggest_match = Disambiguation_tfidf("Danes sem jezdil zebro.", tfidfvectorizer, tfidfvectors)
    print("tfidf result:", data.iloc[biggest_match].meaning)

    # less dependant on corpus vocabulary, slower
    word2vecmodel = word2vec(token_list)
    similarities = Disambiguation_w2v("V Afriki sem jahal zebro.", text_classes, word2vecmodel)
    print(max(similarities, key=similarities.get))