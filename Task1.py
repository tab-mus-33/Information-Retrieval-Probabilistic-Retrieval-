import numpy as np 
import pandas as pd 
import csv 
import re 


from nltk.tokenize import word_tokenize 
from nltk.tokenize import regexp_tokenize 
from nltk.stem import WordNetLemmatizer
from collections import Counter 
import matplotlib.pyplot as plt
 

from os import listdir

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

if __name__ == '__main__':
    
    def generate_dict(list_of_list):
        lemma = WordNetLemmatizer()
        tokens=[]
        #extracting the words from the list by tokenizing them 
        """Only alphabets wetre tokenized to only compare it to Zipf's law for other texts 
        
        Furthermore, this tokenization is done by converting everything into the lowercase first 
        
        """
        for i in range(len(lines)):
            tokens.append([w for w in word_tokenize(lines[i].lower()) if w.isalpha()])
            
        refined=[]
        for sentence in range(len(tokens)):
            for word in range(len(tokens[sentence])):
                refined.append(lemma.lemmatize(tokens[sentence][word]))
        
        return Counter(refined)

    def return_stats(dictionary):
        ranks=np.empty(len(vocabulary))
        freq=np.empty(len(vocabulary), dtype=float)
        terms=[]
        
        sort_in_order=sorted(dictionary.items(), key=lambda x:x[1],reverse=True)
        for rank in range(len(sort_in_order)):
            ranks[rank]=int(rank+1)
            freq[rank]=sort_in_order[rank][1]
            terms.append(sort_in_order[rank][0])
        text_stats=pd.DataFrame(data=[ranks,terms,freq]).T.rename(columns={0:"Rank",1:"Word",2:"Frequency"})
        text_stats['Zipf Frequencies']=(1/text_stats['Rank'])*text_stats['Frequency'][0]
        text_stats["Zipf Fraction"]=text_stats['Rank'].apply(lambda x: "1/{}".format(int(x)))
        text_stats['Normalized Frequencies']=text_stats['Frequency']/text_stats["Frequency"].sum()
        plt.figure(1)
        plt.plot(text_stats["Rank"],text_stats["Frequency"],label="Data trend of text")
        plt.plot(text_stats["Rank"],text_stats["Zipf Frequencies"],label="Zipf Law")
        plt.xlabel('Ranks in Log Scale')
        plt.ylabel('Frequencies in Log Scale')
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Comparison of Text's plot (log(Freq) Vs log(Rank) with Zipf's Law")
        plt.legend()
        plt.savefig("Zipf's Law Comparison.pdf")
        plt.figure(2)
        plt.plot(text_stats["Rank"],text_stats["Normalized Frequencies"])
        plt.xlabel('Rank')
        plt.ylabel('Probabilities of Occurrence')
        plt.title("Probability vs Rank")
        plt.savefig("Probability vs Rank.pdf")
        return text_stats
        
            
    
    lines = []
    with open('passage-collection.txt',encoding='utf-8') as f:
        lines = f.readlines()
        

    generated_dict= generate_dict(lines)
    print("The vocabulary is {} words".format(len(generated_dict)))
        
    print("Saving the vocabulary")
    np.save("vocab_dict", generated_dict) 
    vocabulary=np.load("vocab_dict.npy",allow_pickle=True).item()

    table=return_stats(vocabulary)
    print("Printing the text statistics")
    print(table.head(20))
