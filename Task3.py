import numpy as np 
import pandas as pd 
import csv 
import re 
import time
import json



from collections import Counter 
import matplotlib.pyplot as plt


if __name__ == '__main__':

    import nltk
    from operator import itemgetter

    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
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
    

    
    #total=mp.cpu_count()
    #pool = Pool(total)

    def get_idf(inverted_index):
        idf={}
        for keys in inverted_index:
            check={}
            for keys_sub in inverted_index[keys]:
                check=np.log10(1+(N/len(inverted_index[keys])))
            idf.update({keys:check})
        return idf
    
    def return_tf(inverted_index):
        a={}
        for keys in inverted_index:
            for keys_sub in inverted_index[keys]:
                if keys_sub not in a.keys():
                    a.update({keys_sub:[[keys,inverted_index[keys][keys_sub]]]})
                else:
                    a[keys_sub].append([keys,inverted_index[keys][keys_sub]])
        for keys in a:
            sum=0
            for i in range(len(a[keys])):
                sum+= a[keys][i][-1]
            for i in range(len(a[keys])):
                a[keys][i][-1]=a[keys][i][-1]/sum
        return a

    def tf_idf_passage(term_frequency,idf):
        tf_idf={}
        for keys in term_frequency:
            for i in range(len(term_frequency[keys])):
                if keys not in tf_idf.keys():
                    tf_idf.update({keys:[[term_frequency[keys][i][0],term_frequency[keys][i][-1]*idf[term_frequency[keys][i][0]]]]})
                else:
                    tf_idf[keys].append([term_frequency[keys][i][0],term_frequency[keys][i][-1]*idf[term_frequency[keys][i][0]]])
        return tf_idf

    def vector_space(tf_idf,vocab):
        vector=np.zeros(len(vocab))
        for index in range(len(tf_idf)):
            #for index_array in range(len(vocab)):
            if tf_idf[index][0] in vocab:
                vector[vocab.index(tf_idf[index][0])]=tf_idf[index][-1]
        return vector
    
    def tf_idf_query(term_frequency,idf):
        tf_idf={}
        for keys in term_frequency:
            for i in range(len(term_frequency[keys])):
                if keys not in tf_idf.keys():
                    tf_idf.update({keys:[[term_frequency[keys][i][0],term_frequency[keys][i][-1]*idf[term_frequency[keys][i][0]]]]})
                else:
                    tf_idf[keys].append([term_frequency[keys][i][0],term_frequency[keys][i][-1]*idf[term_frequency[keys][i][0]]])
        return tf_idf
    
    def map_passage(tf_idf_query,passage_test):
        query_to_passage={}
        for i in tf_idf_query.keys():
            query_to_passage.update({i:[]})
        for index in query_to_passage.keys():
            temp=list(passage_test[passage_test["qid"]==index]["pid"])
            for q in range(len(temp)):
                query_to_passage[index].append(temp[q])
        return query_to_passage
    
    def cosine_similarity(tf_idf_query,tf_idf_passage,vocab,query_mapper):
        passage={}
        for i in tf_idf_query.keys():
            if i not in passage.keys():
                passage.update({ i:[]})
        for query_index in tf_idf_query.keys():
            query_vector=vector_space(tf_idf_query[query_index],vocab)
            for pid in range(len(query_mapper[query_index])):
                passage_vector=vector_space(tf_idf_passage[query_mapper[query_index][pid]],vocab)
                cosine_simalarity=np.inner(query_vector,passage_vector)/(np.linalg.norm(query_vector)*np.linalg.norm(passage_vector))
                passage[query_index].append([query_mapper[query_index][pid],cosine_simalarity])
        return passage

    def sort_queries(similarity_score,order):
        fin={}
        for index in range(len(order)):
            similarity_score[order[index]]=sorted(similarity_score[order[index]],key=itemgetter(1),reverse=True)[:100]
            fin.update({order[index]:similarity_score[order[index]]})
        return fin
    
    
    def col_save(sorted_list):
        a=[]
        for keys in sorted_list.keys():
            for i in range(len(sorted_list[keys])):
                a.append([keys,sorted_list[keys][i][0],sorted_list[keys][i][-1]])
        return a
    
    def term_bm_passage(inverted_index):
        a={}
        for keys in inverted_index:
            for keys_sub in inverted_index[keys]:
                if keys_sub not in a.keys():
                    a.update({keys_sub:{keys:inverted_index[keys][keys_sub]}})
                else:
                    a[keys_sub].update({keys:inverted_index[keys][keys_sub]})
        return a

    def term_bm_passage_two(inverted_index):
        a={}
        for keys in inverted_index:
            for keys_sub in inverted_index[keys]:
                if keys_sub not in a.keys():
                    a.update({keys_sub:[[keys,inverted_index[keys][keys_sub]]]})
                else:
                    a[keys_sub].append([keys,inverted_index[keys][keys_sub]])
        return a

    
    
    def document_length(frequency_table):
        length={}
        for keys in frequency_table:
            sum=0
            for i in range(len(frequency_table[keys])):
                sum+= frequency_table[keys][i][-1]
                length.update({keys:sum})
        return length

    def bmidf(inverted_index):
        dictionary={}
        for keys in inverted_index:
            check={}
            for keys_sub in inverted_index[keys]:
                check=np.log(((N-len(inverted_index[keys])+0.5)/(len(inverted_index[keys])+0.5)))
                dictionary.update({keys:check})
        return idf

    def BM25_score(query_x_passage,query_dict,passage_dict,document_lengths,bm_idf):
        k1=1.2
        k2=100
        b=0.75
        avdl=sum(document_lengths.values())/len(document_lengths)
        bm_score={}
        for keys in query_dict.keys():
            if keys not in bm_score.keys():
                bm_score.update({keys:[]})
            for i in range(len(query_x_passage[keys])):
                bm_score_per_query=0
                for q_i in range(len(query_dict[keys])):
                    if query_dict[keys][q_i][0] in passage_dict[query_x_passage[keys][i]].keys():
                        f_i=passage_dict[query_x_passage[keys][i]][query_dict[keys][q_i][0]]
                        qfi=query_dict[keys][q_i][-1]
                        idf=bm_idf[query_dict[keys][q_i][0]]
                        K=k1*(1-b)+((b*document_lengths[query_x_passage[keys][i]])/avdl)
                        bm_score_per_query+=idf*((k1+1)*f_i)/(K+f_i)+((k2+1)*qfi)/(k2+qfi)
                    else:
                        bm_score_per_query+=0
                bm_score[keys].append([query_x_passage[keys][i],bm_score_per_query])
        return bm_score
                
    


    

    
    
    

    passage_test=pd.read_csv("candidate-passages-top1000.tsv",sep='\t',header=None).rename(columns={0:"qid",1:"pid",2:"query",3:"passage"})
    only_unique=passage_test.drop_duplicates(['pid']).reset_index()
    N=len(only_unique["pid"])
    inverted_index=np.load("inv_index_passage.npy",allow_pickle=True).item()
    query_index=np.load("inv_index_query.npy",allow_pickle=True).item()

    with open ("without_stop","r") as file:
        without_stop=json.load(file)
    

    
    print("Processing Term Frequency for passage")
    term_frequency=return_tf(inverted_index)
    print("Processing IDF for passage")
    idf=get_idf(inverted_index)
    print("Processing TFIDF for passage")
    tfidf_passage= tf_idf_passage(term_frequency,idf)
    print("Processing Term Frequency for query")
    tf_query=return_tf(query_index)
    print("Processing tfidf for query")
    tfidf_query= tf_idf_query(tf_query,idf)
    query_to_passage=map_passage(tfidf_query,passage_test)


    
    query_order=pd.read_csv("test-queries.tsv", sep='\t',header=None)

    print("Calculatiing and saving Cosine Similarity")
    
    cosine=cosine_similarity(tfidf_query,tfidf_passage,without_stop,query_to_passage)
    
    query_save_order=list(query_order[0])

    sorted_list=sort_queries(cosine,query_save_order)

    col_map=col_save(sorted_list)

    cosine=pd.DataFrame(col_map,columns=["qid","pid","score"])

    cosine.to_csv("tfidf.csv")
    print("Cosine Similarity saved")

    freq_pas=term_bm_passage(inverted_index)



    freq_que=term_bm_passage(query_index)

    

    freq_pas_list=term_bm_passage_two(inverted_index)

    freq_que_list=term_bm_passage_two(query_index)

    document_lengths=document_length(freq_pas_list)
   
    bm_idf=bmidf(inverted_index)

    print("Calculatiing BM25 Scores")

    bm_scores=BM25_score(query_to_passage,freq_que_list,freq_pas,document_lengths,bm_idf)

    bm_sorted=sort_queries(bm_scores,query_save_order)

    df_list=col_save(bm_sorted)

    bm25=pd.DataFrame(df_list,columns=["qid","pid","score"])
    bm25.to_csv("bm25.csv")
    print("BM25 Sores saved")
    np.save("idf_passage",idf)
    np.save("tf_idf_passage",tfidf_passage)
    np.save("tf_idf_query",tfidf_query)
    np.save("query_to_passage_list",query_to_passage)

    with open("query_save_order","w") as f:
        json.dump(query_save_order,f)

    np.save("document_length_dict",document_lengths)
    np.save("frequency_info_passage",freq_pas)
    np.save("frequency_info_query",freq_que)
    np.save("frequency_info_passage_as_dict_list",freq_pas_list)
    np.save("frequency_info_query_as_dict_query",freq_que_list)



    

    
  

   
    
    
  

    stop=time.time()
    time=stop-go
    print("Total time taken was {} seconds".format(time))

   
