#Find suggestions for ambigous words from corpus using translation slo-eng
import os
from bs4 import BeautifulSoup
import requests

def process_file(contents):
    soup = BeautifulSoup(contents, 'html.parser')
    for row in soup.body.findAll("p"):
        for word in row.findAll("w"):
            #check for for ambiguity of valid lemma's
            lemma = word.attrs.get("lemma")
            if lemma.isalpha() and len(lemma)>2 and lemma.islower():
                r = check_word(lemma)
                if r:
                    ambiguous_words[lemma] = r
                    print(ambiguous_words)

def check_word(w): #check word for ambiguity in dictionary
    url = "https://sl.pons.com/prevod/slovenščina-angleščina/" + w
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    not_found = soup.find("div",{"class": "fuzzysearch"})
    if not_found == None:
        temp = soup.find("span",{"class": "wordclass"})
        if temp != None:
            temp = temp.find("acronym")
            if temp != None:
                word_type = temp.attrs.get("title")
                if word_type == "samostalnik": #or word_type == "glagol":
                    translations = soup.findAll("div",{"class": "translations"})
                    meanings = []
                    for t in translations:
                        data = t.find("h3")
                        if w in data.contents[0] and len(data.contents)>1:
                            meanings.append(data.find("span").contents[0])
                    result = set(meanings)
                    if len(result) > 1:
                        return result
    return False

if __name__ == "__main__":
    ambiguous_words = dict()
    corpus = []
    
    dirname = 'C:/Users/kimbe/Documents/GitHub/cross-lingual-sense-disambiguation/ccGigafidaV1_0'
    for filename in os.listdir(dirname):
        with open(dirname+'/'+filename, encoding='utf8') as file:
            contents = file.read()
            process_file(contents)
