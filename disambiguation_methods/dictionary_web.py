import requests
from bs4 import BeautifulSoup
from googletrans import Translator
from nltk.corpus import wordnet as wn
from util import wordnet_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, download
from util import tokenize, translate_si_en, untag_tokens

download('averaged_perceptron_tagger')

stopwords = stopwords.words('english')
similarities = {}
word_synsets = {}


def translate(sentence):
    translator = Translator()
    translation = translator.translate(sentence, src="sl", dest="en")
    return translation.text


def babel(word):
    # more definitions, but some are very niche
    URL = "https://babelnet.org/search?word=" + word + "&lang=SL&transLang=EN"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    babel_results = soup.find_all("div", {"class": "pos-results"})
    output = {}
    if len(soup.find_all("div", {"class": "synset"})) > 0:  # multiple definitions
        for r in babel_results:
            header = r.find_all("div", {"class": "pos-header"})[0]
            lemma, pos = header.find("span", {"class": "lemma"}).text, header.find("span", {"class": "pos"}).text
            definitions = r.find_all("div", {"class": "synset"})
            output[lemma] = {}
            output[lemma]["pos"] = pos
            output[lemma]["definitions"] = []
            for d in definitions:
                if len(d.find("div", {"class": "meta"}).find_all("span", {"data-type": "NAMED_ENTITY"})) > 0: continue

                # find description
                description = d.find("div", {"class": "definition-by-language"}).find("div",
                                                                                      {"class": "definition"}).text
                description = description.replace("\n", "").replace("\t", "")
                # if description is not in english translate it
                if d.find("div", {"class": "definition-by-language"}).find("span", {
                    "class": "language language-fallback"}) is None:
                    description = translate(description)

                english_word = \
                d.find_all("div", {"class": "synonim-by-language"})[-1].find_all("span", {"class": "synonim"})[0].text
                output[lemma]["definitions"].append(description)
    else:
        dictionary = soup.find("div", {"class": "dictionary"})
        lemma, pos = dictionary.find("span", {"class": "synonim"}).text, dictionary.find("span", {
            "class": "chip chip-primary pos"}).text
        babel_results = soup.find("div", {"class": "translation-by-language"})
        english_word = babel_results.find_all("span", {"class": "synonim"})[0].text
        description = babel_results.find("div", {"class": "definition"}).text.replace("\n", "").replace("\t",
                                                                                                        "").replace(
            "WordNet 3.0", "").strip()
        output[lemma] = {"pos": pos, "definitions": [description]}
    return output


def sskj(word):
    URL = "https://fran.si/iskanje?FilteredDictionaryIds=133&View=1&Query=" + word
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")

    lemma = soup.find("span", {"class": "font_xlarge"}).find("a").text  # will have accents
    output = {}
    results = soup.find("div", {"class": "entry-content"}).find_all("span", {"data-group": "explanation "})
    output[word] = {}
    output[word]["definitions"] = []
    output[word]["definicije"] = []
    for r in results:
        output[word]["definitions"].append(translate(r.text.replace(":", "")))
        output[word]["definicije"].append(r.text.replace(":", ""))
    return output


def similarity(w_list1, w_list2):
    sent_similarity = 0
    global similarities
    for w1 in w_list1:
        w1_s = 0
        for w2 in w_list2: # find the most similar word
            if w1 in w2:
                w1_s = 3.6375861597263857  # max similarity
                if w1 not in similarities.keys(): similarities[w1] = {}
                similarities[w1][w2] = w1_s
                break
            elif w1 in similarities.keys() and w2 in similarities[w1].keys():
                s = similarities[w1][w2]
            elif w2 in similarities.keys() and w1 in similarities[w2].keys():
                s = similarities[w2][w1]
            else:
                s = wordnet_similarity(w1, w2)
                if w1 not in similarities.keys(): similarities[w1] = {}
                similarities[w1][w2] = s
            if s > w1_s: w1_s = s
        sent_similarity += w1_s
    return sent_similarity


def merge_babel_sskj(word):
    # takes a lot of time, WIP
    b = babel(word)[word]["definitions"]
    s = sskj(word)[word]["definitions"]
    smaller = b if len(b) <= len(s) else s
    larger = b if len(b) > len(s) else s
    # calculate similarity to entry in larger
    map_i = {}
    for i, entry1 in enumerate(smaller):
        print(entry1)
        entry1 = [e for e in word_tokenize(entry1) if e not in stopwords]
        entry1 = [e[0].lower() for e in pos_tag(entry1) if e[1].startswith("N")]
        larger_match = [-1, None]  # current largest similarity, match index in larger
        for j, entry2 in enumerate(larger):
            entry2 = [e for e in word_tokenize(entry2) if e not in stopwords]
            entry2 = [e[0].lower() for e in pos_tag(entry2) if e[1].startswith("N")]
            s = similarity(entry1, entry2)
            if s > larger_match[0]: larger_match = [s, j]
        map_i[i] = larger_match
        print(larger_match)
    return map_i


def use_best(query, word):
    #b = babel(word)[word]["definitions"]
    sskj_results = sskj(word)[word]
    #larger = b if len(b) > len(s) else s
    larger = sskj_results["definitions"]
    query = untag_tokens(tokenize([query]))[0]
    query = [translate_si_en(w)[0] for w in query if w != word]
    query = [e for e in word_tokenize(" ".join(query)) if e not in stopwords and e.isalnum()]
    query = [e[0].lower() for e in pos_tag(query)]
    max_match = ["", 0]
    for i, entry in enumerate(larger):
        entry = [e for e in word_tokenize(entry) if e not in stopwords and e.isalnum()]
        entry = [e[0].lower() for e in pos_tag(entry)]
        s = similarity(query, entry)
        # print(query, sskj_results["definitions"][i], sskj_results["definicije"][i], s)
        if s > max_match[1]: max_match = [sskj_results["definicije"][i], s]
    return max_match


# use_best("Boli me jezik.", "jezik")
# use_best("Govorim več jezikov.", "jezik")