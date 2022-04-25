from util import load, tokenize, load_downloaded_model, download_unzip, download_gzip
from tfidf import tfidf, Disambiguation_tfidf
from word2vec import word2vec, Disambiguation_w2v
from LSTM import preprocess_tokens, prepare_data, make_model, metrics
from sklearn.model_selection import train_test_split


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

    # needs a large corpus
    # fname = download_gzip("https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sl.300.vec.gz")
    fname = ""
    word2vecmodel = load_downloaded_model(fname)

    tokenized_sent_list, Y = preprocess_tokens(data)
    X, embedding_matrix, max_length, vocab_size, veclen = prepare_data(data, tokenized_sent_list, word2vecmodel)
    NNmodel = make_model(embedding_matrix, max_length, vocab_size, veclen)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=4)

    history = NNmodel.fit(X, Y, validation_split=0.25, epochs=15, verbose=0)
    metrics(history, NNmodel, X_test, Y_test)







