#%%
import os
import sys
import argparse
import rbo
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from scipy.stats import pearsonr, spearmanr, kendalltau

os.chdir('/root/default/Neural-IR')
sys.path.append(os.getcwd())
print(os.getcwd())

from utils import set_random_seed
from retrieval import *

#%%
def dense_retriever_perturbation(args):
    
    ''' searcher load '''
    faiss_dir = f"./corpus_faiss/ance_{args.corpus_name}_faiss/ance_{args.corpus_name}_faiss"
    faiss_index = faiss.read_index(faiss_dir)

    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()  # use a single GPU
        faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)

    model_base = SentenceBERT("msmarco-roberta-base-ance-firstp", batch_size=128)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load query (train)
    with open(args.query_path,'r') as f:
        query = json.load(f)

    # load actual performance (train)
    with open(args.ap_path, 'r') as r:
        ap_bank = json.loads(r.read())

    # load run (result list), 2 min
    with open(args.run_path, 'rb') as f:
        run = pickle.load(f)

    if args.dataset in ['DL2019', 'DL2020', 'DLHard']:
        args.target_metric = 'ndcg@10'
    elif args.dataset == 'msmarcodev':
        args.target_metric = 'mrr@10'

    pp_list = []
    ap_list = []
    for qid, qtext in tqdm(query.items()):
        
        q_embed = model_base.encode_queries(qtext, show_progress_bar=False, convert_to_tensor=True).to(device='cpu')

        embeddings_sigma = args.noise_std
        # if args.noise_estimate :
        #     embeddings_var = q_embed.var(axis=0)
        # else:
        #     embeddings_var = args.noise_std

        noise_shape = [q_embed.shape[0], args.iter_num]
        noise = torch.normal(mean=0, std=torch.tensor(embeddings_sigma), size=noise_shape)
        q_embed_perturbed = (q_embed.unsqueeze(dim=1) + noise).T.to(torch.float32)

        split_range_num = 10
        assert args.iter_num % split_range_num == 0

        corpus_ids = np.empty((0, args.top_k), dtype=int)
        for j in range(args.iter_num // split_range_num):
            q_embed_perturbed_sub = q_embed_perturbed[split_range_num * j : split_range_num * (j+1)]
            sim, corpus_ids_sub = faiss_index.search(np.array(q_embed_perturbed_sub, order='C'), args.top_k)
            corpus_ids = np.concatenate([corpus_ids, corpus_ids_sub], axis=0)

        pp_array = np.zeros(args.iter_num)
        for i in range(args.iter_num):
            result_list = list(run[qid].keys())[:args.top_k]
            result_list_perturbed = corpus_ids[i, :].astype(str).tolist()

            pp_array[i] = rbo.RankingSimilarity(result_list[:args.top_k], result_list_perturbed[:args.top_k]).rbo()
        
        pp_list.append(pp_array.mean())
        ap_list.append(float(ap_bank[qid][args.target_metric]))

    pearson_coef, pearson_p = pearsonr(ap_list, pp_list)
    kendall_coef, kendall_p = kendalltau(ap_list, pp_list)
    spearman_coef, spearman_p = spearmanr(ap_list, pp_list)
    sMARE, _ = sARE(ap_list, pp_list, rankType='average')
    
    eval_metric_dict = {
        f'pearson_coef_{args.dataset}' : pearson_coef,
        f'kendall_coef_{args.dataset}' : kendall_coef,
        f'spearman_coef_{args.dataset}' : spearman_coef,
        f'sMARE_{args.dataset}' : sMARE,
    }

    return eval_metric_dict


def sARE(ap_list, pp_list, rankType='average'):

    ap_df = pd.Series(ap_list)
    pp_df = pd.Series(pp_list)

    ap_rank = ap_df.rank(method=rankType)
    pp_rank = pp_df.rank(method=rankType)
    sARE_ = np.abs(pp_rank - ap_rank) / pp_rank.shape[0]
    sMARE = sARE_.mean()
    sARE_list = sARE_.tolist()
    
    return sMARE, sARE_list


#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='DenseQPP')
    parser.add_argument("--target_metric", type=str, default='mrr@10')
    parser.add_argument('--corpus_name', type=str, default='msmarco')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument("--base_model", type=str, default='')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--dataset_list", nargs='+', type=str, default='')

    parser.add_argument("--query_path", type=str, default='')
    parser.add_argument("--qrels_path", type=str, default='')
    parser.add_argument("--run_path", type=str, default='')
    parser.add_argument("--ap_path", type=str, default='')
    parser.add_argument("--output_path", type=str, default='')
    parser.add_argument("--index_path", type=str, default='')
    parser.add_argument("--checkpoint_path", type=str, default='')
    parser.add_argument("--setup", type=str, default='')

    #######
    #######
    parser.add_argument("--random_seed", type=int, default=11)
    parser.add_argument('--noise_estimate', type=bool, default=True) 
    parser.add_argument('--noise_std', type=float, default=0.05)
    parser.add_argument('--iter_num', type=int, default=3)
    #######
    #######

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
    
    args.dataset_list = ['msmarcodev', 'DL2019', 'DL2020', 'DLHard']

    eval_metric_all = {}

    base_model = 'bm25'
    dataset = 'msmarcodev'
    dataset = 'DL2020'
    dataset = 'DLHard'

    for base_model in ['bm25', 'ance']:
        args.base_model = base_model

        for dataset in args.dataset_list:
            args.dataset = dataset

            args.query_path = f'./datasets/TREC/{args.dataset}/queries_{args.dataset}.jsonl'
            args.qrels_path = f'./datasets/TREC/{args.dataset}/qrels_{args.dataset}.jsonl'
            args.run_path = f'./retrieval_results/{args.base_model}_{args.dataset}_result'
            args.ap_path = f'./output/actual_performance/{args.base_model}_{args.dataset}_actual_performance.json'
            args.output_path = f'./output/predicted_performance'
            args.index_path = './datasets/collections/lucene-index-msmarco-passage'
            args.checkpoint_path = f'./unsupervisedQPP/{args.name}/checkpoint/'
            args.setup = f"{args.base_model}_{args.dataset}_{args.name}"
            
            set_random_seed(seed=args.random_seed)

            eval_metric_dict = dense_retriever_perturbation(args)

            print(args.base_model)
            print(args.dataset)
            print(eval_metric_dict)
            
            eval_metric_all[f'{args.base_model}_{args.dataset}'] = eval_metric_dict

    print(eval_metric_all)
    eval_metric_all

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
        
    # save result
    with open(f'{args.checkpoint_path}result_DenseQPP.pickle', 'w') as json_file:
        json.dump(eval_metric_all, json_file, indent=4)

    # load result
    with open(f'{args.checkpoint_path}result_DenseQPP.pickle', 'r') as json_file:
        eval_metric_all = json.load(json_file)
