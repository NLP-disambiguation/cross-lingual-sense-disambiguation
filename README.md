# cross-lingual-sense-disambiguation

Go to -> disambiguation.py

Večpomenke
  - zebra - prehod za pešce in žival
  - miška - predmet in žival
  - pajek - žival in vozilo za odvažanje avtomobilov
  - zmaj - žival iz pravljic in predmet
  - jezik - koncept in del telesa
  - list - predmet in del rastline
  - občina - ustanova in upravna enota
  - krilo - oblačilo in del živali

### tokenizer
- slovene tokenizer from classla package
- nltk slovene stopwords
- 
### tfidf
- each tokenized sentence in corpus is vectorized with uni and bigrams
- query is transformed into vector, and then compared with vectors of corpus
- most similar by cosine similarity is chosen

### word2vec
- compare input sentence with sentences in data collection
- break sentence down into words
- for words in each sentence, calculate smooth inverse frequency, SIF with 
similarity then being sum of sif * vector for each word and its vector. 
- this way, all word embeddings can be combined into one
- finally, compare these sentence embeddings with cosine similarity
- if new word is not in model, use translation to english with google translate
(so if a word has more senses, it might translate to one) and take the word in model
that is most sense similar based on Leacock Chodorow method

### Neural network
- used LSTM 
- embeddings from word2vec 
- increase amount of vocabulary by generating similar tokenized sentences with word2vec

