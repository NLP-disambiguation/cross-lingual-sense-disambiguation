import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead
import random
import nltk
from allennlp.modules.elmo import Elmo, batch_to_ids


def load_dataset(path="generated_token_lists.csv"):
    # load generated tokens
    data = pd.read_csv(path)
    data["meaning"] = data["word"] + "_" + data["meaning"]
    n_classes = len(data.meaning.unique())
    train, test = train_test_split(data, test_size=0.25)
    return data, n_classes, train, test


def load_downloaded_model(file_path):
    # load word2vec model
    word2vec = dict()
    i = 0
    with open(file_path, "r", encoding='utf-8') as file:
        file.readline()
        line = file.readline()
        while len(line) > 0:
            if i % 100000 == 0: print("reading...", i)
            i += 1
            try:
                key = line.replace(" \n", "").split(" ", 1)
                key, value = key[0], key[1].split(" ")
                value = np.array([float(val) for val in value], dtype=np.float32)
                word2vec[key] = value
            except Exception as e:
                print(e, line)
            line = file.readline()
    return word2vec, len(word2vec[list(word2vec.keys())[0]])


def w2v_vectorize(df, word2vecmodel, W2VDIM):
    # convert tokens to average vector
    matrix, y = [], []
    for _, r in df.iterrows():
        sent = r.sentence
        sent = sent.split(" ")
        vec = np.zeros((W2VDIM))
        for w in sent:
            try:
                vec += word2vecmodel[w] * (1 / len(sent))
            except:
                print(w, end=", ")
        matrix.append(vec)
        y.append(r.meaning)
    return np.array(matrix), np.array(y)


def do_knn(train_matrix, train_y, test_matrix, test_y, k=3):
    # classify according to kNN
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_matrix, train_y)
    pred = neigh.predict(test_matrix)
    print("knn \n", confusion_matrix(test_y, pred))
    print("knn \n", classification_report(test_y, pred))
    return accuracy_score(test_y, pred)


def one_to_all(train_matrix, train_y, test_matrix, test_y):
    # classify by summing distances to all training examples
    acc = 0
    for i, orig_vec in enumerate(test_matrix):
        dist = {}
        orig_vec = np.squeeze(np.asarray(orig_vec))
        for j, vec in enumerate(train_matrix):
            # same word
            if str(train_y[j].split("_")[0]) != str(test_y[i].split("_")[0]): continue
            vec = np.squeeze(np.asarray(vec))
            d = nltk.cluster.util.cosine_distance(orig_vec, vec)
            if train_y[j] not in dist.keys(): dist[train_y[j]] = 0
            dist[train_y[j]] += d
        # print(testrow.meaning, max(dist, key=dist.get))
        if test_y[i] == min(dist, key=dist.get): acc += 1
    print("1 vs all: ", acc / len(test_matrix))


def do_word2vec(model_path="wiki.sl.vec"):
    data, n_classes, train, test = load_dataset()
    word2vecmodel, W2VDIM = load_downloaded_model(model_path)

    # convert sentences to vector
    train_matrix, train_y = w2v_vectorize(train, word2vecmodel, W2VDIM)
    test_matrix, test_y = w2v_vectorize(test, word2vecmodel, W2VDIM)

    do_knn(train_matrix, train_y, test_matrix, test_y)


def sloberta_vectorize(df, maxlen, tokenizer):
    matrix, y = [], []
    for _, r in df.iterrows():
        sent = r.sentence
        sent = sent.split(" ")
        vec = np.zeros((maxlen))
        for w in sent:
            vec += tokenizer(w, return_tensors="np", padding="max_length", max_length=maxlen)["input_ids"][0] * (1 / len(sent))
        matrix.append(vec)
        y.append(r.meaning)
    return np.array(matrix), np.array(y)


def do_sloberta(modelname="EMBEDDIA/sloberta"):
    data, n_classes, train, test = load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    # model = AutoModelForMaskedLM.from_pretrained(modelname)
    train_matrix, train_y = sloberta_vectorize(train, 20, tokenizer)
    test_matrix, test_y = sloberta_vectorize(test, 20, tokenizer)

    do_knn(train_matrix, train_y, test_matrix, test_y)


def do_gpt2(modelname="macedonizer/sl-gpt2"):
    data, n_classes, train, test = load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    # model = AutoModelWithLMHead.from_pretrained(modelname)
    tokenizer.pad_token = tokenizer.eos_token

    train_matrix, train_y = sloberta_vectorize(train, 20, tokenizer)
    test_matrix, test_y = sloberta_vectorize(test, 20, tokenizer)
    do_knn(train_matrix, train_y, test_matrix, test_y)
    one_to_all(train_matrix, train_y, test_matrix, test_y)


def elmo_vectorize(df, elmo):
    matrix, y = [], []
    for _, r in df.iterrows():
        sent = r.sentence
        sent = sent.split(" ")
        sent_list1 = batch_to_ids([sent])
        vec = elmo(sent_list1)
        vec = vec['elmo_representations'][0].detach().numpy()[0]
        vec = np.sum(vec, axis=0)
        matrix.append(vec)
        y.append(r.meaning)
    return np.array(matrix), np.array(y)


def do_elmo(opt_path="/content/slovenian/options.json", model_path="/content/slovenian/slovenian-elmo-weights.hdf5"):
    # need to download https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1277/slovenian-elmo.tar.gz
    data, n_classes, train, test = load_dataset()
    elmo = Elmo(opt_path, model_path, 1, dropout=0)
    train_matrix, train_y = elmo_vectorize(train, elmo)
    test_matrix, test_y = elmo_vectorize(test, elmo)
    do_knn(train_matrix, train_y, test_matrix, test_y)
