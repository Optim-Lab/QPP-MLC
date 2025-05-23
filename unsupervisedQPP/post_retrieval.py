#%%
import argparse
import easydict
import os
import sys
import pickle
import logging
import json
import faiss
import numpy as np
from scipy.stats import pearsonr

os.chdir('/root/default/qppmlc')
sys.path.append(os.getcwd())
print(os.getcwd())

from utils import set_random_seed

from pyserini.index import IndexReader
from pyserini.search.lucene import LuceneSearcher
from retrieval import *

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#%%
def RM1(pid_list, score_list, index_reader, k, mu=1000):
    V =[]

    if k > len(pid_list):
        k = len(pid_list)
    
    doc_len = np.zeros(k)
    pid_list = np.array(pid_list).astype(str).tolist()

    doc_cache = None
    local_cache = {}
    for idx_p, pid in enumerate(pid_list[:k]):
        if doc_cache and pid in doc_cache:
            doc_vec = doc_cache[pid]
        else:
            doc_vec = index_reader.get_document_vector(pid)
            if doc_cache is not None:
                doc_cache[pid] = doc_vec
        local_cache[pid] = doc_vec
        V += doc_vec.keys()
        doc_len[idx_p] = sum(doc_vec.values())

    V = list(set(V))
    mat = np.zeros([k, len(V)])
    
    token2idx = {token: i for i, token in enumerate(V)}
    for idx_p, pid in enumerate(pid_list[:k]):
        doc_vec = index_reader.get_document_vector(pid)
        for token, freq in doc_vec.items():
            mat[idx_p, token2idx[token]] = freq


    _p_w_q = np.dot(np.array([score_list[:k] / doc_len , ]), mat) # [1, V] become a probability distribution
    p_w_q = np.asarray(_p_w_q/ sum(score_list[:k])).squeeze() # normalisation [V]
    rm1 = np.sort(np.array(list(zip(V, p_w_q)), dtype=[('tokens', object), ('token_scores', np.float32)]), order='token_scores')[::-1] # [V]

    return rm1


def CLARITY(rm1, index_reader, term_num=100):

    rm1_cut = rm1[:term_num] # [term num]
    p_w_q = rm1_cut['token_scores'] / rm1_cut['token_scores'].sum() # make sure it is a probability distribution after sampling
    p_t_D = np.array([[index_reader.get_term_counts(token, analyzer=None)[1] for token in rm1_cut['tokens']], ]) / index_reader.stats()['total_terms'] # [1, term num]
    
    omit_idx = (p_t_D > 0)
    p_t_D = p_t_D[omit_idx]
    p_w_q = p_w_q[omit_idx[0]]

    clarity_score = np.log(p_w_q / p_t_D).dot(p_w_q)

    return clarity_score


def WIG(qtokens, score_list, k):
    corpus_score = np.mean(score_list)
    wig_norm = (np.mean(score_list[:k]) - corpus_score)/ np.sqrt(len(qtokens))
    wig_no_norm = np.mean(score_list[:k]) / np.sqrt(len(qtokens))

    return wig_norm, wig_no_norm


def NQC(score_list, k):
    corpus_score = np.mean(score_list)
    nqc_no_norm = np.std(score_list[:k])
    nqc_norm = nqc_no_norm / corpus_score

    return nqc_norm, nqc_no_norm


def SIGMA_MAX(score_list):
    max_std=0
    scores=[]

    for idx, score in enumerate(score_list):
        scores.append(score)
        if np.std(scores)>max_std:
            max_std = np.std(scores)

    return max_std, len(scores)


def SIGMA_X(qtokens, score_list, x):

    top_score = score_list[0]
    scores = []

    for idx, score in enumerate(score_list):
        if score>=(top_score*x):
            scores.append(score)

    return np.std(scores)/np.sqrt(len(qtokens)), len(scores)


def SMV(score_list, k):
    corpus_score = np.mean(score_list)
    mu = np.mean(score_list[:k])
    smv_norm = np.mean(np.array(score_list[:k])*abs(np.log(score_list[:k]/mu)))/corpus_score
    smv_no_norm = np.mean(np.array(score_list[:k])*abs(np.log(score_list[:k]/mu)))

    return smv_norm, smv_no_norm


def QF(searcher, base_model, model_base, faiss_index, pid_list, rm1, term_num=20, top_k=100):
    
    query_prime = ' '.join(rm1[:term_num]['tokens'])
    
    if base_model == 'bm25':
        ''' BM25 '''
        results = searcher.search(query_prime, top_k) 
        # score = [results[k].score for k in range(len(results))]
        corpus_id = [results[k].docid for k in range(len(results))]

        qf_score = len(set(pid_list[:top_k]) & set(corpus_id)) 

    else:
        if model_base is not None:
            query_embeddings = model_base.encode_queries(query_prime, convert_to_tensor=False, show_progress_bar=False).reshape([1, 768])
            sim, corpus_ids = faiss_index.search(np.array(query_embeddings), top_k)
            qf_score = len(set(pid_list[:top_k]) & set(corpus_ids[0].astype(str)))
        else:
            qf_score = 0

    return qf_score


def UEF_NQC(searcher, base_model, model_base, faiss_index, rm1, score_list, term_num=100, top_k=150):

    corpus_score = np.mean(score_list)
    nqc_no_norm = np.std(score_list[:100])
    nqc_norm = nqc_no_norm / corpus_score

    query_prime = ' '.join(rm1[:term_num]['tokens'])
    
    if base_model == 'bm25':
        ''' BM25 '''
        results = searcher.search(query_prime, top_k) # queries
        score = [results[k].score for k in range(len(results))]
        # corpus_id = [results[k].docid for k in range(len(results))]

        min_len = min(len(score_list), len(score))

        if min_len < top_k:
            uef_score = pearsonr(score_list[:min_len], score[:min_len])[0] * nqc_norm
        else:
            uef_score = pearsonr(score_list[:top_k], score)[0] * nqc_norm

    else:
        if model_base is not None:
            query_embeddings = model_base.encode_queries(query_prime, convert_to_tensor=False, show_progress_bar=False).reshape([1, 768])
            sim, corpus_ids = faiss_index.search(np.array(query_embeddings), top_k)
            uef_score = pearsonr(score_list[:top_k], sim[0])[0] * nqc_norm
        else:
            uef_score = 0

    return uef_score


#%%
def post_retrieval(args, base_model, dataset):

    if not os.path.exists(os.path.abspath(args.output_path)):
        os.makedirs(os.path.abspath(args.output_path))
    
    ''' Load faiss index '''
    if base_model == 'bm25':
        faiss_dir = "./corpus_faiss/{}_{}_faiss/{}_{}_faiss".format('ance', args.corpus_name, 'ance', args.corpus_name)
    else:
        faiss_dir = "./corpus_faiss/{}_{}_faiss/{}_{}_faiss".format(base_model, args.corpus_name, base_model, args.corpus_name)
    
    faiss_index = faiss.read_index(os.path.abspath(faiss_dir)) # 3~10 min
    res = faiss.StandardGpuResources()  # use a single GPU
    faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)

    ''' dense retriever '''
    base_model_dict = {
        'ance' : "msmarco-roberta-base-ance-firstp",
        'dpr' : ("facebook-dpr-question_encoder-multiset-base",
                "facebook-dpr-ctx_encoder-multiset-base",
                " [SEP] "),
    }
    
    if base_model == 'bm25':
        model_base = None
    elif base_model in base_model_dict.keys():
        model_base = SentenceBERT(base_model_dict[base_model], batch_size=128)
    else:
        model_base = None
    
    ''' Load index '''
    searcher = LuceneSearcher(os.path.abspath(args.index_path))
    index_reader = IndexReader(os.path.abspath(args.index_path))

    ''' Load run file '''
    with open(args.run_path, 'rb') as f:
        run = pickle.load(f)
    
    ''' load query file '''
    if dataset == 'DLHard_valid':
        
        dataset_tmp = 'DL2019'
        with open(f'./datasets/TREC/{dataset_tmp}/queries_{dataset_tmp}.jsonl','r') as f:
            query_19 = json.load(f)
        
        dataset_tmp = 'DL2020'
        with open(f'./datasets/TREC/{dataset_tmp}/queries_{dataset_tmp}.jsonl','r') as f:
            query_20 = json.load(f)
        
        dataset_tmp = 'DLHard'
        with open(f'./datasets/TREC/{dataset_tmp}/queries_{dataset_tmp}.jsonl','r') as f:
            query_hard = json.load(f)

        q19 = list(query_19.keys())
        q20 = list(query_20.keys())
        qhard = list(query_hard.keys())

        qhard_diff = (set(q19) | set(q20)) - set(qhard)
        query_all = {**query_19, **query_20}
        queries = {qid: query_all[qid] for qid in qhard_diff if qid in query_all}

    else:
        with open(args.query_path,'r') as f:
            queries = json.load(f)
    
    
    if dataset == 'msmarcotrain':
        queries = dict(list(queries.items())[:5000])
        if list(queries.keys()) != list(run.keys()):
            print("Warning: queries, run differ key order")

    ''' qid, qtext example '''
    print(f'base_model: {base_model}, dataset: {dataset}')
    k_list = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 300, 500, 1000]
    x_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    predicted_performance = {}
    count = 0
    for qid, qtext in queries.items():
        count += 1
        if count == 1 or count % 10 == 0:
            print(f"{count}/{len(queries)}")

        predicted_performance[qid] = {}
        qtokens = index_reader.analyze(qtext)

        pid_list = [pid for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)]
        score_list = [score for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)]

        # top_k = 10
        for top_k in k_list:

            rm1 = RM1(pid_list, score_list, index_reader, k=top_k, mu=args.mu)

            predicted_performance[qid][f"clarity-score-k{top_k}"]= CLARITY(rm1, index_reader, term_num=args.term_num)

            predicted_performance[qid][f"qf-score-k{top_k}"]= QF(searcher, base_model, model_base, faiss_index, pid_list, rm1, term_num=20, top_k=top_k)

            predicted_performance[qid][f"uef-nqc-k{top_k}"]= UEF_NQC(searcher, base_model, model_base, faiss_index, rm1, score_list, term_num=100, top_k=top_k)

            predicted_performance[qid][f"wig-norm-k{top_k}"],  predicted_performance[qid][f"wig-no-norm-{top_k}"] = WIG(qtokens, score_list, top_k)

            predicted_performance[qid][f"nqc-norm-k{top_k}"],  predicted_performance[qid][f"nqc-no-norm-{top_k}"] = NQC(score_list, top_k)

            predicted_performance[qid][f"smv-norm-k{top_k}"],  predicted_performance[qid][f"smv-no-norm-{top_k}"] = SMV(score_list, top_k)

        for x in x_list:

            predicted_performance[qid][f"sigma_x{x}"], actual_k = SIGMA_X(qtokens, score_list, x)

        predicted_performance[qid][f"sigma_max"], actual_k = SIGMA_MAX(score_list)


    name_list = []
    for qid, v in predicted_performance.items():
        name_list = list(v.keys())
        break

    for name in name_list:
        output_path_ = f"{args.output_path}/{base_model}_{dataset}_{name}"
        
        print(f"{name} on the {base_model} dataset")
        print(f"Write predicted performance into the file {output_path_}")

        with open(output_path_, 'w') as pp_w:
            for qid, v in predicted_performance.items():
                pp_w.write(qid + '\t' + str(v[name]) + '\n')

    return None

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--k_clarity", type=int, default=100)
    parser.add_argument("--term_num", type=int, default=100)
    parser.add_argument("--mu", type=int, default=1000)
    parser.add_argument("--k_wig", type=int, default=5)
    parser.add_argument("--k_nqc", type=int, default=100)
    parser.add_argument("--k_smv", type=int, default=100)
    parser.add_argument("--k_uef_nqc", type=int, default=150)
    parser.add_argument("--k_qf", type=int, default=100)
    parser.add_argument("--x", type=float, default=0.5)

    parser.add_argument("--corpus_name", type=str, default='msmarco')

    parser.add_argument("--query_path", type=str, default='')
    parser.add_argument("--run_path", type=str, default='')
    parser.add_argument("--index_path", type=str, default='')
    parser.add_argument("--output_path", type=str, default='')

    parser.add_argument("--base_model_list", nargs='+', type=str, default=[])
    parser.add_argument("--dataset_list", nargs='+', type=str, default=[])

    try:
        args = parser.parse_args()
        print("terminal")
    except:
        args = easydict.EasyDict({
            'corpus_name' : 'msmarco',
            'k_clarity' : 100,
            'k_qf' : 100,
            'k_uef_nqc' : 150,
            'term_num' : 100,
            'mu' : 1000,
            'k_wig' : 5,
            'k_nqc' : 100,
            'k_smv' : 100,
            'x' : 0.5,
            'query_path' : '',
            'run_path' : '',
            'index_path' : '',
            'output_path' : '',
        })
        args = parser.parse_args([])
        print("interactive")

        dataset = 'DL2019'
        base_model = 'ance'
        # dataset = 'msmarcotrain'
        # base_model = 'bm25'
        args.query_path = f'./datasets/TREC/{dataset}/queries_{dataset}.jsonl'
        # args.run_path = f'./retrieval_results/{base_model}_{dataset}_result'
        args.run_path = f'./retrieval_results/{base_model}_{dataset}_result_valid'
        args.index_path = f'./datasets/collections/lucene-index-msmarco-passage'
        args.output_path = f'./output/predicted_performance'

    # set_random_seed(seed=11)
    # post_retrieval(args, base_model, dataset)
        
    set_random_seed(seed=11)
    for base_model in args.base_model_list:
        for dataset in args.dataset_list:

            args.query_path = f'./datasets/TREC/{dataset}/queries_{dataset}.jsonl'
            # args.run_path = f'./retrieval_results/{base_model}_{dataset}_result'
            args.run_path = f'./retrieval_results/{base_model}_{dataset}_result_valid'
            args.index_path = f'./datasets/collections/lucene-index-msmarco-passage'
            args.output_path = f'./output/predicted_performance'
            post_retrieval(args, base_model, dataset)
