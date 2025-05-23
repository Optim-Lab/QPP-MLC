#%%#
import os
import sys
import json
import torch
import argparse
import pickle
import openai 
import pandas as pd
import numpy as np
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau

os.chdir('/root/default/qppmlc')
print(os.getcwd())
sys.path.append(os.getcwd())

#%%
openai.api_key = 'insert_your_api_key'
def get_response(prompt):
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=100
    )

    return response.choices[0].message.content


def create_prompt(query, passage):
    prompt_text = f"""
    Instruction: Please determine whether the given query and document are relevant to each other. Output either ``Relevant'' or ``Irrelevant''.
    Query: {query}
    Passage: {passage}
    Output: Relevant/Irrelevant
    """

    return prompt_text


def create_listwise_prompt(query, document_list):
    prompt_text = f"""
    Instruction: The following documents are retrieved in ranked order (Document 1 is the top-ranked). Assess the relevance of each document to the query. For each document, output "Relevant" or "Irrelevant" based on its content.

    Query: {query}

    """
    for idx, document in enumerate(document_list, 1):
        prompt_text += f"Document {idx} : {document}\n\n"

    prompt_text += "Output (write one per line corresponding to Document 1 to Document {}):\n".format(len(document_list))
    prompt_text += "\n".join([f"Document {i}: Relevant/Irrelevant" for i in range(1, len(document_list)+1)])

    # print(prompt_text)

    return prompt_text


def parse_listwise_response(response: str):
    response_lines = response.strip().split('\n')
    relevance_list = []

    for line in response_lines:
        if "relevant" in line.lower():
            if "irrelevant" in line.lower():
                relevance_list.append(0)  # "Irrelevant"
            else:
                relevance_list.append(1)  # "Relevant" only
        else:
            # print(f"Unexpected line format: '{line}'")
            relevance_list.append(-1)

    return relevance_list


def load_data(args, split=1, method='listwise'):
    searcher = LuceneSearcher(os.path.abspath(args.index_path))

    # load query (train)
    with open(args.query_path,'r') as f:
        query = json.load(f)
    
    # load qrels (train)
    with open(args.qrels_path,'r') as f:
        qrels = json.load(f)

    # load actual performance (train)
    with open(args.ap_path, 'r') as r:
        ap_bank = json.loads(r.read())
    
    # load run (result list), 2 min
    with open(args.run_path, 'rb') as f:
        run = pickle.load(f)

    len(query)
    len(run)

    ''' checkpoint save '''
    # split = 1
    weight_dir = f"{args.checkpoint_path}listwise/"
    weight_path = f"{weight_dir}result_{args.setup}_{split}.json"
    
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    
    # Try loading existing results if they exist
    if os.path.exists(weight_path):
        with open(weight_path, "r") as f:
            result_dict = json.load(f)
    else:
        result_dict = {}
    
    num_iter = 0
    # qid = list(run.keys())[2]
    for qid in tqdm(run.keys()):
        
        num_iter += 1
        
        if qid not in qrels:
            print(f"skip {qid}")
            continue

        if len(run[qid]) < 10:
            print(f"skip {qid} : result list less then top_k")
            continue
        
        # Skip already processed queries
        if qid in result_dict:
            continue
        
        query_i = query[qid]
        pid_k = [pid for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)[:args.top_k]]
        doc_top_k = [json.loads(searcher.doc(str(pid)).raw())['contents'] for pid in pid_k]

        if method == 'pointwise':
            qid_dict = {}
            for j in range(len(doc_top_k)):
                
                doc_j = doc_top_k[j]
                prompt = create_prompt(query_i, doc_j)
                # response = get_response(prompt)
                response = 'Irrelevant'

                if response in ["Relevant", "relevant"]:
                    response = 1
                elif response in ["Irrelevant", "irrelevant"]:
                    response = 0
                else:
                    print(f'response is {response}.')
                    response = -1

                qid_dict[pid_k[j]] = response

            result_dict[qid] = qid_dict

            if num_iter % 100 == 0:
                with open(weight_path, "w") as f:
                    json.dump(result_dict, f, indent=4, sort_keys=True)
        
        elif method == 'listwise':
            
            prompt = create_listwise_prompt(question=query_i, document_list=doc_top_k)
            response = get_response(prompt)
            # print(response)

            binary_labels = parse_listwise_response(response)
            qid_dict = {pid: label for pid, label in zip(pid_k, binary_labels)}

            result_dict[qid] = qid_dict

            if num_iter % 100 == 0 or num_iter == len(run.keys()):
                with open(weight_path, "w") as f:
                    # json.dump(result_dict, f, indent=4, sort_keys=True)
                    json.dump(result_dict, f, indent=4, sort_keys=False)

    return result_dict


def compute_batch_rr(predicted_relevance):
    predicted_relevance_partial = predicted_relevance
    predicted_relevance_binary = predicted_relevance_partial

    # Find the first occurrence of 1 in each row (along the top_m dimension)
    positions = torch.arange(1, 10 + 1, device=predicted_relevance.device).float()

    # Compute reciprocal rank for the first 1 in each row
    rr_pred = (predicted_relevance_binary / positions).max(dim=0).values.item()
    
    return rr_pred


def compute_batch_dcg(predicted_relevance):
    # Compute DCG
    discounts = torch.log2(torch.arange(2, 10 + 2).float().to(predicted_relevance.device))

    gains = predicted_relevance
    dcg_pred = (gains / discounts).sum(dim=-1).item()

    return dcg_pred


def compute_batch_idcg(predicted_relevance):
    # Compute DCG
    discounts = torch.log2(torch.arange(2, 10 + 2).float().to(predicted_relevance.device))

    # Compute ideal DCG (IDCG) by sorting relevance ideally (highest scores first)
    ideal_relevance, _ = torch.sort(predicted_relevance, dim=-1, descending=True)
    ideal_gains = ideal_relevance
    idcg_pred = (ideal_gains / discounts).sum(dim=-1).item()
    
    return idcg_pred


def sARE(ap_list, pp_list, rankType='average'):

    ap_df = pd.Series(ap_list)
    pp_df = pd.Series(pp_list)

    ap_rank = ap_df.rank(method=rankType)
    pp_rank = pp_df.rank(method=rankType)
    sARE_ = np.abs(pp_rank - ap_rank) / pp_rank.shape[0]
    sMARE = sARE_.mean()
    sARE_list = sARE_.tolist()
    
    return sMARE, sARE_list


def QPP_gpt_result(args, subset=False):
    ''' checkpoint save '''
    split = 1
    # weight_dir = f"{args.checkpoint_path}pointwise/"
    weight_dir = f"{args.checkpoint_path}listwise/"
    weight_path = f"{weight_dir}result_{args.setup}_{split}.json"
    
    # search
    searcher = LuceneSearcher(os.path.abspath(args.index_path))

    with open(weight_path, "r") as f:
        result_dict = json.load(f)
    
    # load query (train)
    with open(args.query_path,'r') as f:
        query = json.load(f)
    
    # load qrels (train)
    with open(args.qrels_path,'r') as f:
        qrels = json.load(f)
    
    # load run (result list), 2 min
    with open(args.run_path, 'rb') as f:
        run = pickle.load(f)
    
    # load actual performance (train)
    with open(args.ap_path, 'r') as r:
        ap_bank = json.loads(r.read())

    if args.dataset in ['DL2019', 'DL2020', 'DLHard']:
        args.target_metric = 'ndcg@10'
    elif args.dataset == 'msmarcodev':
        args.target_metric = 'mrr@10'

    qid_list = list(run.keys())
    
    if subset:
        num_q = 50
        qid_list = qid_list[:num_q]
    else:
        num_q = len(qid_list)
    
    pp_list = []
    ap_list = []

    predicted_relevance_list = torch.zeros((len(qid_list), args.top_k))
    actual_relevance_list = torch.zeros((len(qid_list), args.top_k))
    score_list = torch.zeros((len(qid_list), args.top_k))

    rr_true_list = []
    dcg_true_list = []
    idcg_true_list = []
    ndcg_true_list = []

    rr_pred_list = []
    dcg_pred_list = []
    idcg_pred_list = []
    ndcg_pred_list = []

    ndcg_tilde_list = []
    
    i=0
    for i in range(num_q):
        qid = qid_list[i]
        
        # prediction
        predicted_relevance = torch.tensor(list(result_dict[qid_list[i]].values()))
        predicted_relevance_list[i, :] = predicted_relevance

        rr_pred = compute_batch_rr(predicted_relevance)
        dcg_pred = compute_batch_dcg(predicted_relevance)
        idcg_pred = compute_batch_idcg(predicted_relevance)
        ndcg_pred = dcg_pred / idcg_pred if idcg_pred > 0 else 0

        pid_k, score = zip(*[(pid, score) for pid, score in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)[:args.top_k]])
        score = torch.tensor(score, dtype=float)

        score_list[i, :] = score

        qrels_posi = {k: v for k, v in qrels[qid].items() if v > 0}
        relevance_grades = torch.tensor([qrels_posi.get(key, 0) for key in pid_k])
        
        doc_top_k = [json.loads(searcher.doc(str(pid)).raw())['contents'] for pid in pid_k]
        query[qid]
        doc_top_k
        
        # Compute relevance array
        run_pids = list(run[qid].keys())[:args.top_k]
        rel_array = np.zeros(args.top_k)  # Initialize with zeros
        rel_array[:len(run_pids)] = [qrels_posi.get(pid, 0) for pid in run_pids]

        # Compute ideal relevance array
        sorted_items = sorted(qrels_posi.values(), reverse=True)
        rel_array_ideal = np.zeros(args.top_k)
        rel_array_ideal[:len(sorted_items[:args.top_k])] = sorted_items[:args.top_k]

        # Compute DCG and IDCG
        discounts = np.log2(np.arange(2, args.top_k + 2))
        dcg_true = (rel_array / discounts).sum()
        idcg_true = (rel_array_ideal / discounts).sum()
        
        # Compute nDCG@m
        ndcg_true = dcg_true / idcg_true if idcg_true > 0 else 0

        # Compute rr@m
        positions = torch.arange(1, args.top_k + 1).float()
        if args.dataset in ['msmarcotrain', 'msmarcodev']:
            relevance_binary = relevance_grades[:args.top_k] >= 1
        else:
            relevance_binary = relevance_grades[:args.top_k] >= 2

        rr_true = (relevance_binary / positions).max()
        
        actual_relevance_list[i, :] = relevance_grades

        # rr_true = compute_batch_rr(relevance_grades)
        ndcg_tilde = dcg_pred / idcg_true if idcg_true > 0 else 0

        rr_true_list.append(rr_true)
        dcg_true_list.append(dcg_true)
        idcg_true_list.append(idcg_true)
        ndcg_true_list.append(ndcg_true)

        rr_pred_list.append(rr_pred)
        dcg_pred_list.append(dcg_pred)
        idcg_pred_list.append(idcg_pred)
        ndcg_pred_list.append(ndcg_pred)
        
        ndcg_tilde_list.append(ndcg_tilde)
        
        if idcg_pred == 0:
            ndcg_pred = 0
        else:
            ndcg_pred = dcg_pred / idcg_pred

        ndcg = ap_bank[qid_list[i]]['ndcg@10']
        rr = ap_bank[qid_list[i]]['mrr@10']

        if args.target_metric == 'mrr@10':
            pp_list.append(rr_pred)
            ap_list.append(rr)
        elif args.target_metric == 'ndcg@10':
            # pp_list.append(ndcg_pred)
            pp_list.append(dcg_pred)
            ap_list.append(ndcg)

    # pp_list = torch.rand(16).cuda().tolist()
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

    dict_embed = {
        # 'embed' : embed,
        'qid_list' : qid_list,
        'predicted_relevance' : predicted_relevance_list,
        'actual_relevance' : actual_relevance_list,
        'score' : score_list,
        'rr_pred' : torch.tensor(rr_pred_list),
        'dcg_pred' : torch.tensor(dcg_pred_list),
        'idcg_pred' : torch.tensor(idcg_pred_list),
        'ndcg_pred' : torch.tensor(ndcg_pred_list),
        'rr_true' : torch.tensor(rr_true_list),
        'dcg_true' : torch.tensor(dcg_true_list),
        'idcg_true' : torch.tensor(idcg_true_list),
        'ndcg_true' : torch.tensor(ndcg_true_list),
        'ndcg_tilde' : torch.tensor(ndcg_tilde_list),
    }
    
    return eval_metric_dict, ap_list, pp_list


#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='gpt_4o')
    parser.add_argument("--mode", type=str, default='pilot')
    parser.add_argument("--target_metric", type=str, default='mrr@10')
    parser.add_argument("--base_model", type=str, default='')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--query_path", type=str, default='')
    parser.add_argument("--qrels_path", type=str, default='')
    parser.add_argument("--run_path", type=str, default='')
    parser.add_argument("--ap_path", type=str, default='')
    parser.add_argument("--index_path", type=str, default='')
    parser.add_argument("--setup", type=str, default='')
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--data_chunk_size", type=int, default=10000)
    parser.add_argument("--num_split", type=int, default=51)
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embed_model", type=str, default='bert_cross')
    
    try:
        print("terminal")
        args = parser.parse_args()
    except:
        print("interactive")
        args = parser.parse_args([])
        args.base_model = 'bm25'
        args.dataset = 'msmarcodev'
        args.dataset_list = ['DL2019', 'DL2020', 'DLHard']
        args.dataset_list = ['msmarcodev', 'DL2019', 'DL2020', 'DLHard']

        args.top_k = 10

        args.query_path = f'./datasets/TREC/{args.dataset}/queries_{args.dataset}.jsonl'
        args.qrels_path = f'./datasets/TREC/{args.dataset}/qrels_{args.dataset}.jsonl'
        args.run_path = f'./retrieval_results/{args.base_model}_{args.dataset}_result'
        args.ap_path = f'./output/actual_performance/{args.base_model}_{args.dataset}_actual_performance.json'
        args.index_path = './datasets/collections/lucene-index-msmarco-passage'
        args.checkpoint_path = f'./supervisedQPP/gpt_4o/checkpoint/'
        args.setup = f"{args.base_model}_{args.dataset}_{args.name}"

    ''' DL2019, DL2020, DLHard '''
    for split in [1, 2, 3]:
        for base_model in ['bm25', 'ance']:
            for dataset in ['msmarcodev', 'DL2019', 'DL2020', 'DLHard']:
                
                print(f'split: {split}, base_model: {base_model}, dataset: {dataset}')
                
                args.base_model = base_model
                args.dataset = dataset

                args.query_path = f'./datasets/TREC/{args.dataset}/queries_{args.dataset}.jsonl'
                args.qrels_path = f'./datasets/TREC/{args.dataset}/qrels_{args.dataset}.jsonl'
                args.run_path = f'./retrieval_results/{args.base_model}_{args.dataset}_result'
                args.ap_path = f'./output/actual_performance/{args.base_model}_{args.dataset}_actual_performance.json'
                args.index_path = './datasets/collections/lucene-index-msmarco-passage'
                args.checkpoint_path = f'./supervisedQPP/gpt_4o/checkpoint/'
                args.setup = f"{args.base_model}_{args.dataset}_{args.name}"

                result_dict = load_data(args, split=split)

    ''' get result from json file '''
    for base_model in ['bm25', 'ance']:
        for dataset in ['msmarcodev', 'DL2019', 'DL2020', 'DLHard']:
            
            print(f'split: 1, base_model: {base_model}, dataset: {dataset}')
            
            args.base_model = base_model
            args.dataset = dataset

            args.query_path = f'./datasets/TREC/{args.dataset}/queries_{args.dataset}.jsonl'
            args.qrels_path = f'./datasets/TREC/{args.dataset}/qrels_{args.dataset}.jsonl'
            args.run_path = f'./retrieval_results/{args.base_model}_{args.dataset}_result'
            args.ap_path = f'./output/actual_performance/{args.base_model}_{args.dataset}_actual_performance.json'
            args.index_path = './datasets/collections/lucene-index-msmarco-passage'
            args.checkpoint_path = f'./supervisedQPP/gpt_4o/checkpoint/'
            args.setup = f"{args.base_model}_{args.dataset}_{args.name}"

            eval_metric_dict, ap_list, pp_list = QPP_gpt_result(args)
            print(eval_metric_dict)
