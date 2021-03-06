from util import load, tokenize, load_downloaded_model, download_unzip, download_gzip
from disambiguation_methods.tfidf import tfidf, Disambiguation_tfidf
from disambiguation_methods.word2vec import word2vec, Disambiguation_w2v, generate_additional_data, load_keyed_vector_model
from disambiguation_methods.LSTM import preprocess_tokens, prepare_data, make_model, metrics
from disambiguation_methods.dictionary_web import use_best
from sklearn.model_selection import train_test_split
from corpus_augmentation.text_generation import LSTM_generation


def do_LSTM(data):
    # download_unzip('https://hdl.handle.net/1839/a627f276-6d9f-4c8c-9259-11b0d1141dc7@download')
    fname = "wiki.sl.vec"

    generate = False
    if generate: tokenized_sent_list, Y = LSTM_generation(data, fname)
    else:
        tokenized_sent_list, Y, Y_original = preprocess_tokens(data)
    word2vecmodel = load_downloaded_model(fname)
    X, embedding_matrix, max_length, vocab_size, veclen = prepare_data(tokenized_sent_list, word2vecmodel)
    NNmodel = make_model(embedding_matrix, max_length, vocab_size, veclen, len(set(Y)))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=4)

    history = NNmodel.fit(X_train, Y_train, validation_split=0.30, epochs=25, verbose=0)
    metrics(history, NNmodel, X_test, Y_test)


def do_tfidf(token_list, data):
    tfidfvectorizer, tfidfvectors, top_ngrams = tfidf(token_list)
    matches, biggest_match = Disambiguation_tfidf("Danes sem jezdil zebro.", tfidfvectorizer, tfidfvectors)
    print("tfidf result:", data.iloc[biggest_match].meaning)


def do_word2vec(token_list, text_classes):
    word2vecmodel = word2vec(token_list)
    similarities = Disambiguation_w2v("V Afriki sem jahal zebro.", text_classes, word2vecmodel)
    print(max(similarities, key=similarities.get))


def do_dictionary(data):
    with open("dictionary_disambiguation.txt", "w") as f:
        f.write("sentence, word, definition, LCH score")
        for i, r in data.iterrows():
            res = use_best(r.sentence, r.word)
            print(res)
            f.write(",".join([r.sentence, r.word, res[0], str(res[1])]) + "\n")


if __name__ == "__main__":
    # WARNING for googletrans and gensim - update versions
    # pip install --upgrade gensim to v4 if error
    # pip install googletrans==3.1.0a0
    # google colab: sys.path.append("<path to code folder>")

    data, text_classes = load('data.csv')
    token_list = tokenize(list(data.sentence.values))

    #
    # needs a large corpus to do well
    #
    do_LSTM(data)

    #
    # relies a corpus matches, faster
    #
    do_tfidf(token_list, data)

    #
    # less dependant on corpus vocabulary, slower
    #
    do_word2vec(token_list, text_classes)

    #
    # unsupervised, dependant on exhaustive dictionary definitions
    #
    do_dictionary(data)

