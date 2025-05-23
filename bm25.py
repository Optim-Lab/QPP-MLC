#%%
import os
import sys
import argparse
import easydict
import gc
import json
import pathlib
import pickle
import math
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

os.chdir('/root/default/qppmlc')
sys.path.append(os.getcwd())
print(os.getcwd())

#%%
def BM25_pyserini(args, valid=False):
    searcher = LuceneSearcher(os.path.abspath(args.index_path))

    if args.dataset == 'DLHard_valid':
        
        dataset_tmp = 'DL2019'
        args.query_path = f'./datasets/TREC/{dataset_tmp}/queries_{dataset_tmp}.jsonl'
        with open(args.query_path,'r') as f:
            query_19 = json.load(f)
        
        dataset_tmp = 'DL2020'
        args.query_path = f'./datasets/TREC/{dataset_tmp}/queries_{dataset_tmp}.jsonl'
        with open(args.query_path,'r') as f:
            query_20 = json.load(f)
        
        dataset_tmp = 'DLHard'
        args.query_path = f'./datasets/TREC/{dataset_tmp}/queries_{dataset_tmp}.jsonl'
        with open(args.query_path,'r') as f:
            query_hard = json.load(f)

        q19 = list(query_19.keys())
        q20 = list(query_20.keys())
        qhard = list(query_hard.keys())

        qhard_diff = (set(q19) | set(q20)) - set(qhard)
        query_all = {**query_19, **query_20}
        query = {qid: query_all[qid] for qid in qhard_diff if qid in query_all}

    else:
        with open(args.query_path,'r') as f:
            query = json.load(f)

    query_list = [query[qid] for qid in query]
    query_ids = list(query.keys())

    results = {qid: {} for qid in query_ids}

    split_num = 1000
    iter_num = int(len(query_list) / split_num) + 1
    for i in tqdm(range(iter_num)):

        query_list_sub = query_list[split_num * i : split_num * (i+1)]
        query_ids_sub = query_ids[split_num * i : split_num * (i+1)]
        len(query_list_sub)
        len(query_ids_sub)
        
        results_ = searcher.batch_search(query_list_sub, query_ids_sub, k=args.top_k, threads=16)

        count = 0
        for i, qid in enumerate(query_ids_sub):
            
            count += 1
            score = [results_[qid][k].score for k in range(len(results_[qid]))]
            corpus_id = [results_[qid][k].docid for k in range(len(results_[qid]))]
            
            score_dict = dict(zip(corpus_id, score))
            results[qid] = score_dict

    run_dir = f'./retrieval_results'

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    if valid:
        with open(args.run_path + '_valid', 'wb') as f:
            pickle.dump(results, f)
    
    else:
        with open(args.run_path, 'wb') as f:
            pickle.dump(results, f)
    
    return results


def BM25_pyserini_train(index_path, base_model, dataset, top_k=100, query_chunk_size=50000, valid=False):
    searcher = LuceneSearcher(index_path)

    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets/TREC/")

    with open('{}{}/queries_{}.jsonl'.format(out_dir, dataset, dataset),'r') as f:
        queries = json.load(f)
    
    if valid:
        queries = dict(list(queries.items())[:5000])

    results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "retrieval_results")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    queries_list = [queries[qid] for qid in queries]
    query_ids = list(queries.keys())

    itr = range(0, len(queries_list), query_chunk_size)
    start_idx = 0
    for start_idx in itr:
        
        end_idx = min(start_idx + query_chunk_size, len(queries_list))

        sub_queries_list = queries_list[start_idx:end_idx]
        sub_query_ids = query_ids[start_idx:end_idx]

        sub_results = {qid: {} for qid in sub_query_ids}
        sub_results_ = searcher.batch_search(sub_queries_list, sub_query_ids, k=top_k, threads=64)
        
        for i, qid in enumerate(sub_query_ids):
            
            score = [sub_results_[qid][k].score for k in range(len(sub_results_[qid]))]
            corpus_id = [sub_results_[qid][k].docid for k in range(len(sub_results_[qid]))]
            
            score_dict = dict(zip(corpus_id, score))
            sub_results[qid] = score_dict

        num_split = int(end_idx / query_chunk_size)
        if end_idx == len(queries_list):
            num_split = int(math.ceil(end_idx / query_chunk_size))
        
        num_split = str(num_split).zfill(2)

        if valid:
            results_file = "{}/{}_{}_result_valid".format(results_dir, base_model, dataset)
        else:
            results_file = "{}/{}_{}_result_{}".format(results_dir, base_model, dataset, num_split)
        
        with open(results_file, 'wb') as f:
            pickle.dump(sub_results, f)
        
        print('Save {} file'.format(results_file.split('/')[-1]))
        sub_results.clear()
        gc.collect()
    
    return None


#%%
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--index_path', type=str, default='datasets/collections/lucene-index-msmarco-passage')
    parser.add_argument('--top_k', type=int, default=1000)
    parser.add_argument("--query_chunk_size", type=int, default=50000)
    parser.add_argument('--base_model', type=str, default='')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument('--base_model_list', nargs='+', type=str, required=True)
    parser.add_argument("--dataset_list", nargs='+', type=str, required=True)
    
    try:
        args = parser.parse_args()
    except:
        args = easydict.EasyDict({
            'index_path' : 'datasets/collections/lucene-index-msmarco-passage',
            'top_k' : 1000,
            'query_chunk_size' : 50000,
            'base_model' : '',
            'dataset' : '',
            'base_model_list' : '',
            'dataset_list' : '',
        })
        args.base_model_list = ['bm25']
        args.dataset_list = ['msmarcodev', 'DL2019', 'DL2020', 'DLHard']
        args.base_model = 'bm25'
        args.dataset = 'DL2019'
        args.dataset = 'msmarcotrain'

    
    for base_model in args.base_model_list:
        args.base_model = base_model

        for dataset in args.dataset_list:
            args.dataset = dataset
            args.query_path = f'./datasets/TREC/{args.dataset}/queries_{args.dataset}.jsonl'
            args.index_path = './datasets/collections/lucene-index-msmarco-passage'
            args.run_path = f'./retrieval_results/{args.base_model}_{args.dataset}_result'
            results = BM25_pyserini(args)
    
    if dataset == 'msmarcotrain':
        for base_model in args.base_model_list:
            results = BM25_pyserini_train(index_path=args.index_path,
                                          base_model=base_model,
                                          dataset=dataset,
                                          top_k=args.top_k,
                                          query_chunk_size=args.query_chunk_size)
    else:
        for base_model in args.base_model_list:
            for dataset in args.dataset_list:
                results = BM25_pyserini(index_path=args.index_path,
                                        base_model=base_model,
                                        dataset=dataset,
                                        top_k=args.top_k)


    ## generate validation dataset
    # for base_model in ['bm25']:
    #     args.base_model = base_model

    #     for dataset in ['DL2019', 'DL2020', 'DLHard', 'DLHard_valid', 'msmarcodev']:
    #         args.dataset = dataset
    #         args.query_path = f'./datasets/TREC/{args.dataset}/queries_{args.dataset}.jsonl'
    #         args.index_path = './datasets/collections/lucene-index-msmarco-passage'
    #         args.run_path = f'./retrieval_results/{args.base_model}_{args.dataset}_result'
    #         results = BM25_pyserini(args, valid=True)
        
    #     args.dataset = 'msmarcotrain'
    #     args.query_path = f'./datasets/TREC/{args.dataset}/queries_{args.dataset}.jsonl'
    #     args.index_path = './datasets/collections/lucene-index-msmarco-passage'
    #     args.run_path = f'./retrieval_results/{args.base_model}_{args.dataset}_result'
    #     results = BM25_pyserini_train(index_path=args.index_path,
    #                             base_model=args.base_model,
    #                             dataset=args.dataset,
    #                             top_k=args.top_k,
    #                             valid=True
    #                             )
        