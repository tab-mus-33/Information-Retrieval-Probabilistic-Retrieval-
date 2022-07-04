import numpy as np 
import pandas as pd 

import time
import json
from operator import itemgetter





if __name__ == '__main__':



    go=time.time()
    def sort_queries(similarity_score,order):
        fin={}
        for index in range(len(order)):
            similarity_score[order[index]]=sorted(similarity_score[order[index]],key=itemgetter(1),reverse=True)[:100]
            fin.update({order[index]:similarity_score[order[index]]})
        return fin
    
    def save_helper(sorted_list):
        a=[]
        for keys in sorted_list.keys():
            for i in range(len(sorted_list[keys])):
                a.append([keys,sorted_list[keys][i][0],sorted_list[keys][i][-1]])
        return a
    
    def laplace_smoothing(query_x_passage,query_dict,passage_dict,document_lengths,without_stop):
        V=len(without_stop)
        laplace_score={}
        for keys in query_dict.keys():
            if keys not in laplace_score.keys():
                laplace_score.update({keys:[]})
            for i in range(len(query_x_passage[keys])):
                lp_per_query=1
                for q_i in range(len(query_dict[keys])):
                    if query_dict[keys][q_i][0] in passage_dict[query_x_passage[keys][i]].keys():
                        m_i=passage_dict[query_x_passage[keys][i]][query_dict[keys][q_i][0]]
                        D=document_lengths[query_x_passage[keys][i]]
                        lp_per_query*=(m_i+1)/(D+V)
                    else:
                        lp_per_query*=1/(D+V)
                laplace_score[keys].append([query_x_passage[keys][i],np.log(lp_per_query)])
        return laplace_score

    def lidstone_correction(query_x_passage,query_dict,passage_dict,document_lengths,without_stop):
        V=len(without_stop)
        epsilon=0.1
        lindstone_score={}
        for keys in query_dict.keys():
            if keys not in lindstone_score.keys():
                lindstone_score.update({keys:[]})
            for i in range(len(query_x_passage[keys])):
                ln_per_query=1
                for q_i in range(len(query_dict[keys])):
                    if query_dict[keys][q_i][0] in passage_dict[query_x_passage[keys][i]].keys():
                        m_i=passage_dict[query_x_passage[keys][i]][query_dict[keys][q_i][0]]
                        D=document_lengths[query_x_passage[keys][i]]
                        ln_per_query*=(m_i+epsilon)/(D+(epsilon*V))
                    else:
                        ln_per_query*=epsilon/(D+(epsilon*V))
                lindstone_score[keys].append([query_x_passage[keys][i],np.log(ln_per_query)])
        return lindstone_score

    def dirichlet_smoothing(query_x_passage,query_dict,passage_dict,document_lengths,iv_passage):
        total_word_sum=0
        for keys in freq_pas.keys():
            total_word_sum+=sum(freq_pas[keys].values())
        mu=50
        
        dr_score={}
        for keys in query_dict.keys():
            if keys not in dr_score.keys():
                dr_score.update({keys:[]})
            for i in range(len(query_x_passage[keys])):
                dr_per_query=0
                for q_i in range(len(query_dict[keys])):
                    if query_dict[keys][q_i][0] in passage_dict[query_x_passage[keys][i]].keys():
                        N=document_lengths[query_x_passage[keys][i]]
                        lamd=N/(N+mu)
                        one_minus_lamd=mu/(N+mu)
                        p_w_d=(passage_dict[query_x_passage[keys][i]][query_dict[keys][q_i][0]])/N
                        doc_count=len(iv_passage[query_dict[keys][q_i][0]])
                        p_w_c=doc_count/total_word_sum
                        dr_per_query+=np.log((lamd*p_w_d)+(one_minus_lamd*p_w_c))
                    else:
                        doc_count=len(iv_passage[query_dict[keys][q_i][0]])
                        p_w_c=doc_count/total_word_sum
                        dr_per_query+=np.log(p_w_c)
                dr_score[keys].append([query_x_passage[keys][i],dr_per_query])
        return dr_score
                

    
  
    passage_test=pd.read_csv("candidate-passages-top1000.tsv",sep='\t',header=None).rename(columns={0:"qid",1:"pid",2:"query",3:"passage"})
    only_unique=passage_test.drop_duplicates(['pid']).reset_index()
    N=len(only_unique["pid"])
    inverted_index=np.load("inv_index_passage.npy",allow_pickle=True).item()
    query_to_passage=np.load("query_to_passage_list.npy",allow_pickle=True).item()
    freq_que_list=np.load("frequency_info_query_as_dict_query.npy",allow_pickle=True).item()
    freq_pas=np.load("frequency_info_passage.npy",allow_pickle=True).item()
    document_lengths=np.load("document_length_dict.npy",allow_pickle=True).item()

    with open("query_save_order","r") as file:
        query_save_order=json.load(file)
    
    
    with open ("without_stop","r") as file:
        without_stop=json.load(file)


    

    lp_scores=laplace_smoothing(query_to_passage,freq_que_list,freq_pas,document_lengths,without_stop)


    lp_sorted=sort_queries(lp_scores,query_save_order)

    lp_list=save_helper(lp_sorted)

    lp=pd.DataFrame(lp_list,columns=["qid","pid","score"])

    lp.to_csv("laplace.csv")


    ln_scores=lidstone_correction(query_to_passage,freq_que_list,freq_pas,document_lengths,without_stop)

    ln_sorted=sort_queries(ln_scores,query_save_order)

    ln_list=save_helper(ln_sorted)

    ln=pd.DataFrame(ln_list,columns=["qid","pid","score"])

    ln.to_csv("lidstone.csv")

    dr_scores=dirichlet_smoothing(query_to_passage,freq_que_list,freq_pas,document_lengths,inverted_index)

    dr_sorted=sort_queries(dr_scores,query_save_order)

    dr_list=save_helper(dr_sorted)

    dr=pd.DataFrame(dr_list,columns=["qid","pid","score"])

    dr.to_csv("dirichlet.csv")

    stop=time.time()

    stop=time.time()
    time=stop-go
    print("Total time taken was {} seconds".format(time))


    






   

   