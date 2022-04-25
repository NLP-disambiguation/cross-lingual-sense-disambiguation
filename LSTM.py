from util import tokenize, untag_tokens
import pandas as pd
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy import zeros
from keras.layers import Embedding, Input, Flatten, Bidirectional, LSTM, TimeDistributed, Dense
from keras import Model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def preprocess_tokens(data, source_column = "sentence", target_column = "meaning"):
    token_list = untag_tokens(tokenize(" ".join(list(data[source_column].values))))
    token_list = pd.Series(token_list).apply(lambda x: " ".join(x))

    le = preprocessing.LabelEncoder()
    Y_new = data[target_column]
    Y_new = le.fit_transform(Y_new)

    return token_list, Y_new

def prepare_data(data, tokenized_sent_list, w2vmodel):
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(tokenized_sent_list)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(data['sentence'].values)
    # pad documents to a max length of 10 words
    max_length = 10
    X = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


    veclen = w2vmodel['telefon'].shape[0]
    embedding_matrix = zeros((vocab_size, veclen))
    for word, i in t.word_index.items():
        embedding_vector = w2vmodel.get(word)
        if embedding_vector is not None:
            try:
                embedding_matrix[i] = embedding_vector
            except:
                print(word)
    return X, embedding_matrix, max_length, vocab_size, veclen

def make_model(embedding_matrix, max_len, vocab_size, veclen):
    input = Input(shape=(max_len,))
    model = Embedding(vocab_size, veclen, weights=[embedding_matrix], input_length=max_len)(input)
    model = Bidirectional(LSTM(veclen, return_sequences=True, dropout=0.50), merge_mode='concat')(model)
    model = TimeDistributed(Dense(veclen, activation='relu'))(model)
    model = Flatten()(model)
    model = Dense(100, activation='relu')(model)
    output = Dense(3, activation='softmax')(model)
    model = Model(input, output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def metrics(history, model, X_test, Y_test):
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=2)
    print('Accuracy: %f' % (accuracy * 100))

    Y_pred = model.predict(X_test)
    y_pred = np.array([np.argmax(pred) for pred in Y_pred])
    print('Classification Report:\n', classification_report(Y_test, y_pred), '\n')

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
