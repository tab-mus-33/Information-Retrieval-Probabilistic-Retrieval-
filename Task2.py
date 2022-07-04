import numpy as np 
import pandas as pd 
import csv 
import re 
import time



from collections import Counter 
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing as mp
from multiprocessing import Process
import json 

if __name__ == '__main__':

    import nltk
    #nltk.download('punkt')
    #nltk.download('wordnet')
    #nltk.download('stopwords')
    from nltk.corpus import stopwords
    import nltk
    from nltk.tokenize import word_tokenize 
    from nltk.tokenize import regexp_tokenize 
    from nltk.stem import WordNetLemmatizer
    
    from spacy.lang.en.stop_words import STOP_WORDS
    lemma = WordNetLemmatizer()
    from_nltk = stopwords.words('english')
    combined_corpus=set.union(set(from_nltk),STOP_WORDS)
    go=time.time()

    vocabulary=np.load("vocab_dict.npy",allow_pickle=True).item()

    
    total=mp.cpu_count()
    pool = Pool(total)
    
    def remove_stop(dictionary,stopwords):
        words=list(dictionary.keys())
        removed=[tokens for tokens in words if tokens not in stopwords]
        return removed
    
    without_stop=remove_stop(vocabulary,combined_corpus)

    
    passage=pd.read_csv("candidate-passages-top1000.tsv",sep='\t',header=None).rename(columns={0:"qid",1:"pid",2:"query",3:"passage"})


    

    def convert_passage(text):
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize 
        from nltk.stem import WordNetLemmatizer
        from spacy.lang.en.stop_words import STOP_WORDS
        lemma = WordNetLemmatizer()
        from_nltk = stopwords.words('english')
        combined_corpus=set.union(set(from_nltk),STOP_WORDS)
        
        text_cleaned = list(set(lemma.lemmatize(w) for w in word_tokenize(text.lower()) if w.isalpha() and w not in combined_corpus))
        return text_cleaned
    
    def query_inverted_index(dataframe,vocabulary):
        only_unique=dataframe.drop_duplicates(['qid']).reset_index()
        for i in range(len(only_unique)):
            only_unique['query'][i]=[lemma.lemmatize(w) for w in word_tokenize(only_unique['query'][i].lower()) if w.isalpha() and w not in combined_corpus]
        
        inverted_dict={}
        for i in range(len(vocabulary)):
            if vocabulary[i] not in inverted_dict.keys():
                inverted_dict.update({ vocabulary[i]:[]})
        #First creates a list object for each word as a key, that is storing passage ids
        for index in range(len(only_unique)):
            for word in only_unique['query'][index]:
                    if word in inverted_dict.keys():
                        inverted_dict[word].append(only_unique['qid'][index])
                        
        #Then counts term frequency using a counter object 
        for keys in inverted_dict:
            inverted_dict[keys]=dict(Counter(inverted_dict[keys]))
        return inverted_dict
    
    def index_invert(dataframe,vocabulary):
        inverted_dict={}
        for i in range(len(vocabulary)):
            if vocabulary[i] not in inverted_dict.keys():
                inverted_dict.update({ vocabulary[i]:[]})
        only_unique=dataframe.drop_duplicates(['pid']).reset_index()
        #First creates a list object for each word as a key, that is storing passage ids
        for index in range(len(only_unique)):
            for word in only_unique['passage'][index]:
                    if word in inverted_dict.keys():
                        inverted_dict[word].append(only_unique['pid'][index])
                        
        #Then counts term frequency using a counter object 
        for keys in inverted_dict:
            inverted_dict[keys]=dict(Counter(inverted_dict[keys]))
        return inverted_dict
    
    passage['passage'] = pool.map(convert_passage, passage['passage'])
    

    
    inverted_index=index_invert(passage,without_stop)

    query_index=query_inverted_index(passage,without_stop)

    np.save("inv_index_passage", inverted_index) 
    np.save("inv_index_query",query_index)
    
    with open("without_stop","w") as f:
        json.dump(without_stop,f)


    
    
  

    stop=time.time()

    print("Time taken was {} seconds".format(stop-go))

    pool.close()
    pool.join()
    pool.clear()



    


    