#Prepare corpus from ccGigafida corpus using known ambigious words
import os
from bs4 import BeautifulSoup
import re

def process_file(contents, ambiguous_words):
    soup = BeautifulSoup(contents, 'html.parser')
    for row in soup.body.findAll("p"):
        b = False
        for word in row.findAll("w"):
            #check for for ambiguity
            lemma = word.attrs.get("lemma")
            for i in ambiguous_words:
                if lemma == i:
                    #zapiši cel stavek v corpus
                    s = re.sub(CLEANR, '', str(row))
                    s = ' '.join(s.splitlines())
                    s = re.sub(' +', ' ', s)
                    corpus.append([s,lemma])
                    b = True
                if b:
                    break
                    
if __name__ == "__main__":
    ambiguous_words = ["zebra","miška","pajek","zmaj","jezik","list","občina","krilo","mesec","revija","točka","čelo","sapa","fant"]
    corpus = []
    CLEANR = re.compile('<.*?>') 

    cnt = 0
    dirname = 'C:/Users/kimbe/Documents/GitHub/cross-lingual-sense-disambiguation/ccGigafidaV1_0'
    for filename in os.listdir(dirname):
        with open(dirname+'/'+filename, encoding='utf8') as file:
            contents = file.read()
            process_file(contents, ambiguous_words)
            if cnt>100: #first 100 files for safety
                break

    f = open("corpus.csv", "w", encoding='utf8')
    f.write('sentence,meaning,word\n')
    for el in corpus:
        f.write(el[0]+',,'+el[1]+'\n')
    f.close()
