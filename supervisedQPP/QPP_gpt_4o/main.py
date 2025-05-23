#%%#
import os
import sys
import json
import torch
import argparse
import pickle
import copy
import tiktoken
import openai 
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from adjustText import adjust_text
from pyserini.search.lucene import LuceneSearcher
from collections import Counter
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
# warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

from scipy.stats import pearsonr, spearmanr, kendalltau

import numpy as np
from sklearn.metrics.cluster import contingency_matrix, adjusted_rand_score, normalized_mutual_info_score

os.chdir('/root/default/Neural-IR')
print(os.getcwd())
sys.path.append(os.getcwd())

#%%
openai.api_key = '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'
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

# question = query_i
# document_list = doc_top_k
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
                relevance_list.append(0)  # "Irrelevant" 포함
            else:
                relevance_list.append(1)  # "Relevant" only
        else:
            # print(f"Unexpected line format: '{line}'")
            relevance_list.append(-1)  # 예외 처리용

    return relevance_list


# def QPPGPT(docs, seedset):
#     seedwords = {f'set {i+1}': seedset.T.iloc[i].tolist() for i in range(len(seedset.T))}
#     assignment = []

#     for i, document in enumerate(tqdm(docs, desc="Processing documents")):
#         try:
#             # truncated_doc = truncating(document)
#             prompt = create_prompt(seedwords, document)
#             response = get_response(prompt)
#             assignment.append(response)
#         except Exception as e:
#             print(f"Error at document {i}: {e}")

#     return assignment

# question = 'How to watch tv?'
# passage = 'apple tv is good'
# prompt = create_prompt(question, passage)
# response = get_response(prompt)

#%%
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
            # response = get_response(prompt)
            response = 'Document 1: Relevant  \nDocument 2: Relevant  \nDocument 3: Relevant  \nDocument 4: Irrelevant  \nDocument 5: Relevant  \nDocument 6: Relevant  \nDocument 7: Relevant  \nDocument 8: Relevant  \nDocument 9: Relevant  \nDocument 10: Relevant'
            # print(response)

            binary_labels = parse_listwise_response(response)
            qid_dict = {pid: label for pid, label in zip(pid_k, binary_labels)}

            result_dict[qid] = qid_dict

            if num_iter % 100 == 0 or num_iter == len(run.keys()):
                with open(weight_path, "w") as f:
                    # json.dump(result_dict, f, indent=4, sort_keys=True)
                    json.dump(result_dict, f, indent=4, sort_keys=False)

    return result_dict
# 12*43+2
# 12*54+2
# 12*6980+2

def compute_batch_rr(predicted_relevance):

    # if self.args.threshold_optimal:
    #     threshold_optimal = torch.tensor([0.061, 0.043, 0.034, 0.025, 0.011, 0.017, 0.019, 0.013, 0.008, 0.007], device=predicted_relevance.device)
    #     threshold_optimal = threshold_optimal[:self.min_k_m]
    #     predicted_relevance_binary = (predicted_relevance_partial > threshold_optimal).int()
    # else:
    #     predicted_relevance_binary = (predicted_relevance_partial > self.args.threshold).int()
    
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

        # true
        # if qid not in qrels:
        #     print(f"skip {qid}")
        #     continue

        # if len(run[qid]) < args.top_k:
        #     print(f"skip {qid} : result list less then top_k")
        #     continue

        pid_k, score = zip(*[(pid, score) for pid, score in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)[:args.top_k]])
        score = torch.tensor(score, dtype=float)

        score_list[i, :] = score

        qrels_posi = {k: v for k, v in qrels[qid].items() if v > 0}
        relevance_grades = torch.tensor([qrels_posi.get(key, 0) for key in pid_k])
        
        doc_top_k = [json.loads(searcher.doc(str(pid)).raw())['contents'] for pid in pid_k]
        query[qid]
        doc_top_k
        
        ###########
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

        ###########
        # Compute rr@m
        positions = torch.arange(1, args.top_k + 1).float()
        if args.dataset in ['msmarcotrain', 'msmarcodev']:
            relevance_binary = relevance_grades[:args.top_k] >= 1
        else:
            relevance_binary = relevance_grades[:args.top_k] >= 2

        rr_true = (relevance_binary / positions).max()
        ###########
        
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
        ##############
        ##############
        ##############
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
def eda_gpt_4o(args, dataset_name):
    
    # dataset_name = 'msmarcodev'
    # dataset_name = 'DL2019'
    # dataset_name = 'DL2020'
    # dataset_name = 'DLHard'
    args_infer = copy.deepcopy(args)
    args_infer.dataset = dataset_name
    if dataset_name in ['DL2019', 'DL2020', 'DLHard']:
        args_infer.target_metric = 'ndcg@10'
    
    args_infer.run_path = f'./retrieval_results/{args_infer.base_model}_{args_infer.dataset}_result'
    args_infer.qrels_path = f'./datasets/TREC/{args_infer.dataset}/qrels_{args_infer.dataset}.jsonl'
    args_infer.query_path = f'./datasets/TREC/{args_infer.dataset}/queries_{args_infer.dataset}.jsonl'
    args_infer.ap_path = f'./output/actual_performance/{args_infer.base_model}_{args_infer.dataset}_actual_performance.json'
    args_infer.setup = f"{args_infer.base_model}_{args_infer.dataset}_{args_infer.name}"
    args_infer.setup_dataset = f"{args_infer.base_model}_{args_infer.dataset}_{args_infer.target_metric}"
    args_infer.batch_size = int(32)

    if dataset_name == 'msmarcodev':
        subset = True
    else:
        subset = False
    
    dict_embed, ap_list, pp_list = QPP_gpt_result(args_infer, subset=subset)
    dict_embed.keys()
    dict_embed['qid_list']

    #############
    #############

    save_dir = f"./output_fig/{args.name}_{args.base_model}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df, df_pred, df_true, df_score = generate_df(dict_embed)
    num_q = dict_embed['score'].shape[0]

    save_path = os.path.join(save_dir, f"df_{dataset_name}.csv")
    df.to_csv(save_path)
    
    ''' scatter plot for metric '''
    # rr@10
    fig_rr, rr_pear_coef, rr_kend_coef = scatter_metric(
        pred=df['rr_pred'].iloc[:num_q],
        true=df['rr_true'].iloc[:num_q],
        model_name=args_infer.name,
        retriever=args_infer.base_model,
        dataset_name=dataset_name,
        metric='rr@10',
        query_ids=None,
        show_plot=True
        )
    
    # ndcg@10
    fig_ndcg, ndcg_pear_coef, ndcg_kend_coef = scatter_metric(
        pred=df['ndcg_pred'].iloc[:num_q],
        true=df['ndcg_true'].iloc[:num_q],
        model_name=args_infer.name,
        retriever=args_infer.base_model,
        dataset_name=dataset_name,
        metric='ndcg@10',
        query_ids=None,
        show_plot=True
        )

    # dcg@10
    fig_dcg, dcg_pear_coef, dcg_kend_coef = scatter_metric(
        pred=df['dcg_pred'].iloc[:num_q],
        true=df['dcg_true'].iloc[:num_q],
        model_name=args_infer.name,
        retriever=args_infer.base_model,
        dataset_name=dataset_name,
        metric='dcg@10',
        query_ids=None,
        show_plot=True
        )
    
    # idcg@10
    fig_idcg, idcg_pear_coef, idcg_kend_coef = scatter_metric(
        pred=df['idcg_pred'].iloc[:num_q],
        true=df['idcg_true'].iloc[:num_q],
        model_name=args_infer.name,
        retriever=args_infer.base_model,
        dataset_name=dataset_name,
        metric='idcg@10',
        query_ids=None,
        show_plot=True
        )

    # ndcg@10 tilde
    fig_ndcg_tilde, ndcg_tilde_pear_coef, ndcg_tilde_kend_coef = scatter_metric(
        pred=df['ndcg_tilde'].iloc[:num_q],
        true=df['ndcg_true'].iloc[:num_q],
        model_name=args_infer.name,
        retriever=args_infer.base_model,
        dataset_name=dataset_name,
        metric='ndcg@10_tilde',
        query_ids=None,
        show_plot=True
        )

    fig_list_scatter = [fig_rr, fig_ndcg, fig_dcg, fig_idcg, fig_ndcg_tilde]

    cols = 3
    rows = 2
    fig_scale = 8
    fig_scatter, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))

    # Figure 배치
    for idx, ax in enumerate(axes.flatten()):
        if idx < len(fig_list_scatter):
            fig_list_scatter[idx].canvas.draw()  # Figure 렌더링
            img = np.array(fig_list_scatter[idx].canvas.renderer.buffer_rgba())  # RGBA 변환
            ax.imshow(img)  # 이미지로 표시
            # ax.set_title(f"Query {idx+1}")
        ax.axis("off")  # 축 제거

    # 전체 레이아웃 조정
    plt.tight_layout()
    plt.show()
    
    save_path = os.path.join(save_dir, f"scatter_{dataset_name}_merge.png")
    fig_scatter.savefig(save_path, dpi=300, bbox_inches="tight")  # 이미지 저장

    ''' bar plot '''
    fig_bar_list = plot_bar(df_pred, df_true, df_score, dataset_name, show_plot=False)
    fig_bar_list[0]
    # plt.scatter([1,2],[2,3])

    # 8×6 Grid에 배치하여 표시
    cols = 4
    rows = (len(fig_bar_list) - 1) // cols + 1

    fig_scale = 8
    fig_bar, axes = plt.subplots(rows, cols, figsize=(cols * fig_scale * 1.5, rows * fig_scale))

    # Figure 배치
    for idx, ax in enumerate(axes.flatten()):
        if idx < len(fig_bar_list):
            fig_bar_list[idx].canvas.draw()  # Figure 렌더링
            img = np.array(fig_bar_list[idx].canvas.renderer.buffer_rgba())  # RGBA 변환
            ax.imshow(img)  # 이미지로 표시
            # ax.set_title(f"Query {idx+1}")
        ax.axis("off")  # 축 제거

    # 전체 레이아웃 조정
    plt.tight_layout()
    plt.show()
    
    save_path = os.path.join(save_dir, f"bar_{dataset_name}.png")
    fig_bar.savefig(save_path, dpi=300, bbox_inches="tight")  # 이미지 저장

    return None


def scatter_metric(pred, true, model_name, retriever, dataset_name, metric, query_ids=None, show_plot=False):

    pear_coef, _ = pearsonr(pred, true)
    kend_coef, _ = kendalltau(pred, true)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pred, true, alpha=0.7, color="royalblue", edgecolors="black", s=70)
    
    if query_ids is None:
        query_ids = np.arange(1, len(pred) + 1)

    # 텍스트 객체 리스트 (adjustText에 사용)
    text_objects = []
    
    for i, (x, y) in enumerate(zip(pred, true)):
        txt = ax.text(x, y, str(query_ids[i]), fontsize=10, color="black")
        text_objects.append(txt)  # 텍스트 객체 저장

    # adjustText 적용 (텍스트 자동 위치 조정)
    adjust_text(text_objects, expand=(3, 2), arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_title(f'{model_name}, {retriever}, {dataset_name}, {metric}', fontsize=14)
    ax.set_xlabel(f'{metric}_pred', fontsize=18)
    ax.set_ylabel(f'{metric}_true', fontsize=18)

    if metric == 'rr@10' or metric == 'ndcg@10':
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])

    text_str = f"Pearson: {pear_coef:.2f}\nKendall: {kend_coef:.2f}"
    ax.text(
        0.95, 0.05, text_str, transform=plt.gca().transAxes,
        fontsize=18, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
    )

    if show_plot:
        plt.show()
    # else:
    #     plt.close(fig)
    
    return fig, pear_coef, kend_coef


def plot_bar(df_pred, df_true, df_score, dataset_name, show_plot=False):
    
    if dataset_name=='msmarcodev':
        pred_scale = 1
    else:
        pred_scale = 1

    score_scale = 0.25
    true_color = "royalblue"   # True 값 색상
    pred_color = "limegreen"  # Pred 값 색상
    score_color = "darkorange"  # score 값 색상

    fig_list = []
    num_q = df_pred.shape[0] - 3
    i = 0
    for i in range(num_q):
        # i번째 query에 대한 true & pred 값 (각 10개)
        true_scores = df_true.iloc[i, :10].values
        pred_scores = df_pred.iloc[i, :10].values
        scores = df_score.iloc[i, :10].values

        # rr, ndcg 값 가져오기
        rr_true_i = df_true['rr_true'].iloc[i]
        dcg_true_i = df_true['dcg_true'].iloc[i]
        ndcg_true_i = df_true['ndcg_true'].iloc[i]
        rr_pred_i = df_pred['rr_pred'].iloc[i]
        dcg_pred_i = df_pred['dcg_pred'].iloc[i]
        ndcg_pred_i = df_pred['ndcg_pred'].iloc[i]

        # X축 레이블 (R1 ~ R10)
        x_labels = [f"R{j+1}" for j in range(10)]
        x = np.arange(len(x_labels))  # X 위치
        width = 0.3  # bar 너비

        # 플롯 생성
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width, true_scores, width, label="True", color=true_color, alpha=0.7)
        ax.bar(x, pred_scores * pred_scale, width, label="Pred", color=pred_color, alpha=0.7)
        ax.bar(x + width, scores * score_scale, width, label="score", color=score_color, alpha=0.7)

        # 범례에 rr, ndcg 값 추가
        legend_text = [
            f"True: RR={rr_true_i:.2f}, NDCG={ndcg_true_i:.2f}, DCG={dcg_true_i:.2f}",
            f"Pred: RR={rr_pred_i:.2f}, NDCG={ndcg_pred_i:.2f}, DCG={dcg_pred_i:.2f}",
            f"Score: Mean={scores.mean():.2f}, Std={scores.std():.2f}",
        ]
        ax.legend(legend_text)

        # 제목 및 라벨 설정
        if dataset_name == "msmarcodev":
            ax.set_ylim((-1.0, 1.1))
        else:
            ax.set_ylim((-1.0, 3.3))
        
        ax.set_xlabel("Documents (R1 - R10)")
        ax.set_ylabel("Relevance Score")
        ax.set_title(f"Query {i+1}: True vs Predicted Relevance")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

        # 그래프 출력
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        fig_list.append(fig)

    return fig_list


def generate_df(dict_embed):
    ''' relevance '''
    # shape : (n=50, top_k=10)
    rel_true = dict_embed['actual_relevance'].cpu()
    rel_pred = dict_embed['predicted_relevance'].cpu()
    rel_pred.shape

    ''' metric, (rr, dcg, idcg, ndcg, ndcg_tilde) '''
    # shape : (n=50)
    rr_true = dict_embed['rr_true'].cpu()
    dcg_true = dict_embed['dcg_true'].cpu()
    idcg_true = dict_embed['idcg_true'].cpu()
    ndcg_true = dict_embed['ndcg_true'].cpu()

    # shape : (n=50)
    rr_pred = dict_embed['rr_pred'].cpu()
    dcg_pred = dict_embed['dcg_pred'].cpu()
    idcg_pred = dict_embed['idcg_pred'].cpu()
    ndcg_pred = dict_embed['ndcg_pred'].cpu()

    # shape : (n=50)
    ndcg_tilde = dict_embed['ndcg_tilde'].cpu()

    # shape : (n=50)
    score = dict_embed['score'].cpu()


    ''' prediction dataframe '''
    df_pred = torch.cat((
        rel_pred,
        rel_pred.mean(dim=1).unsqueeze(1),
        rel_pred.std(dim=1).unsqueeze(1),
        rr_pred.unsqueeze(1),
        dcg_pred.unsqueeze(1),
        idcg_pred.unsqueeze(1),
        ndcg_pred.unsqueeze(1),
        ndcg_tilde.unsqueeze(1),
        ), dim=-1)
    df_pred = pd.DataFrame(df_pred)
    
    df_pred_mean = df_pred.mean().to_frame().T
    df_pred_std = df_pred.std().to_frame().T
    df_pred_max = df_pred.max().to_frame().T
    df_pred = pd.concat([df_pred, df_pred_mean, df_pred_std, df_pred_max])

    df_index_pred = [f"q{i+1}" for i in range(rel_true.shape[0])] + ["mean", "std", "max"]
    df_columns_pred = [f"R{i+1}_pred" for i in range(10)] + ["R_mean_pred", "R_std_pred"] + ["rr_pred", "dcg_pred", "idcg_pred", "ndcg_pred", "ndcg_tilde"]

    # 인덱스와 컬럼 이름 변경
    df_pred.index = df_index_pred
    df_pred.columns = df_columns_pred

    ''' true dataframe '''
    df_true = torch.cat((
        rel_true,
        rel_true.float().mean(dim=1).unsqueeze(1),
        rel_true.float().std(dim=1).unsqueeze(1),
        rr_true.unsqueeze(1),
        dcg_true.unsqueeze(1),
        idcg_true.unsqueeze(1),
        ndcg_true.unsqueeze(1),
        ), dim=-1)
    df_true = pd.DataFrame(df_true)
    
    df_true_mean = df_true.mean().to_frame().T
    df_true_std = df_true.std().to_frame().T
    df_true_max = df_true.max().to_frame().T
    df_true = pd.concat([df_true, df_true_mean, df_true_std, df_true_max])

    df_index_true = [f"q{i+1}" for i in range(rel_true.shape[0])] + ["mean", "std", "max"]
    df_columns_true = [f"R{i+1}_true" for i in range(10)] + ["R_mean_true", "R_std_true"]+ ["rr_true", "dcg_true", "idcg_true", "ndcg_true"]

    # 인덱스와 컬럼 이름 변경
    df_true.index = df_index_true
    df_true.columns = df_columns_true

    ''' score '''
    df_score = torch.cat((
        score,
        # score.mean(dim=1).unsqueeze(1),
        # score.std(dim=1).unsqueeze(1),
        ), dim=-1)
    df_score = pd.DataFrame(df_score)

    df_score_mean = df_score.iloc[:, :10].values.mean()
    df_score_std = df_score.iloc[:, :10].values.std()
    df_score_max = df_score.iloc[:, :10].max()

    df_score = (df_score - df_score_mean) / (df_score_std)

    df_score = pd.concat([df_score, df_score.mean(axis=1), df_score.std(axis=1)], axis=1)

    df_score_mean = df_score.mean().to_frame().T
    df_score_std = df_score.std().to_frame().T
    df_score_max = df_score.max().to_frame().T
    df_score = pd.concat([df_score, df_score_mean, df_score_std, df_score_max])

    df_index_score = [f"q{i+1}" for i in range(rel_true.shape[0])] + ["mean", "std", "max"]
    # df_columns_score = [f"s{i+1}" for i in range(10)]
    df_columns_score = [f"s{i+1}" for i in range(10)] + ["s_mean", "s_std"]

    df_score.index = df_index_score
    df_score.columns = df_columns_score

    ''' df merge '''
    df = pd.concat([df_pred, df_true, df_score], axis=1)
    df.shape

    return df, df_pred, df_true, df_score

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
        # args.base_model = 'ance'
        # args.dataset = 'DL2019'
        # args.dataset = 'DL2020'
        # args.dataset = 'DLHard'
        # args.dataset = 'msmarcotrain'
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

        # eval_metric_dict, ap_list, pp_list = QPP_gpt_result(args, subset=False)
        # eval_metric_dict

    ''' ndcg_check ''' 
    # eda_gpt_4o(args, dataset_name='DL2019')

    # for base_model in ['bm25', 'ance']:
    #     args.base_model = base_model

    #     if args.base_model == 'ance':
    #         args.dataset_list = ['DL2019', 'DL2020', 'DLHard']
    #     else:
    #         args.dataset_list = ['msmarcodev', 'DL2019', 'DL2020', 'DLHard']
        
    #     for dataset_name in args.dataset_list:
            
    #         args.query_path = f'./datasets/TREC/{args.dataset}/queries_{args.dataset}.jsonl'
    #         args.qrels_path = f'./datasets/TREC/{args.dataset}/qrels_{args.dataset}.jsonl'
    #         args.run_path = f'./retrieval_results/{args.base_model}_{args.dataset}_result'
    #         args.ap_path = f'./output/actual_performance/{args.base_model}_{args.dataset}_actual_performance.json'
    #         args.index_path = './datasets/collections/lucene-index-msmarco-passage'
    #         args.checkpoint_path = f'./supervisedQPP/gpt_4o/checkpoint/'
    #         args.setup = f"{args.base_model}_{args.dataset}_{args.name}"
            
    #         eda_gpt_4o(args, dataset_name=dataset_name)


    ''' DL2019, DL2020, DLHard '''
    # for split in [1]:
    #     for base_model in ['bm25', 'ance']:
    #         for dataset in ['DL2019', 'DL2020', 'DLHard']:
                
    #             print(f'split: {split}, base_model: {base_model}, dataset: {dataset}')
                
    #             args.base_model = base_model
    #             args.dataset = dataset

    #             args.query_path = f'./datasets/TREC/{args.dataset}/queries_{args.dataset}.jsonl'
    #             args.qrels_path = f'./datasets/TREC/{args.dataset}/qrels_{args.dataset}.jsonl'
    #             args.run_path = f'./retrieval_results/{args.base_model}_{args.dataset}_result'
    #             args.ap_path = f'./output/actual_performance/{args.base_model}_{args.dataset}_actual_performance.json'
    #             args.index_path = './datasets/collections/lucene-index-msmarco-passage'
    #             args.checkpoint_path = f'./supervisedQPP/gpt_4o/checkpoint/'
    #             args.setup = f"{args.base_model}_{args.dataset}_{args.name}"

    #             result_dict = load_data(args, split=split)

    # ''' msmarcodev '''
    # args.top_k = 10
    # for split in [1]:
    #     for base_model in ['bm25', 'ance']:
    #         for dataset in ['msmarcodev']:
                
    #             print(f'split: {split}, base_model: {base_model}, dataset: {dataset}')
                
    #             args.base_model = base_model
    #             args.dataset = dataset

    #             args.query_path = f'./datasets/TREC/{args.dataset}/queries_{args.dataset}.jsonl'
    #             args.qrels_path = f'./datasets/TREC/{args.dataset}/qrels_{args.dataset}.jsonl'
    #             args.run_path = f'./retrieval_results/{args.base_model}_{args.dataset}_result'
    #             args.ap_path = f'./output/actual_performance/{args.base_model}_{args.dataset}_actual_performance.json'
    #             args.index_path = './datasets/collections/lucene-index-msmarco-passage'
    #             args.checkpoint_path = f'./supervisedQPP/gpt_4o/checkpoint/'
    #             args.setup = f"{args.base_model}_{args.dataset}_{args.name}"

    #             result_dict = load_data(args, split=split)
    

    ''' msmarcodev '''
    args.top_k = 10
    split = 1
    
    print(f'split: {split}, base_model: {args.base_model}, dataset: {args.dataset}')

    args.query_path = f'./datasets/TREC/{args.dataset}/queries_{args.dataset}.jsonl'
    args.qrels_path = f'./datasets/TREC/{args.dataset}/qrels_{args.dataset}.jsonl'
    args.run_path = f'./retrieval_results/{args.base_model}_{args.dataset}_result'
    args.ap_path = f'./output/actual_performance/{args.base_model}_{args.dataset}_actual_performance.json'
    args.index_path = './datasets/collections/lucene-index-msmarco-passage'
    args.checkpoint_path = f'./supervisedQPP/gpt_4o/checkpoint/'
    args.setup = f"{args.base_model}_{args.dataset}_{args.name}"

    result_dict = load_data(args, split=split)

    ''' get result from json file '''
    # for base_model in ['bm25', 'ance']:
    #     for dataset in ['DL2019', 'DL2020', 'DLHard']:
            
    #         print(f'split: 1, base_model: {base_model}, dataset: {dataset}')
            
    #         args.base_model = base_model
    #         args.dataset = dataset

    #         args.query_path = f'./datasets/TREC/{args.dataset}/queries_{args.dataset}.jsonl'
    #         args.qrels_path = f'./datasets/TREC/{args.dataset}/qrels_{args.dataset}.jsonl'
    #         args.run_path = f'./retrieval_results/{args.base_model}_{args.dataset}_result'
    #         args.ap_path = f'./output/actual_performance/{args.base_model}_{args.dataset}_actual_performance.json'
    #         args.index_path = './datasets/collections/lucene-index-msmarco-passage'
    #         args.checkpoint_path = f'./supervisedQPP/gpt_4o/checkpoint/'
    #         args.setup = f"{args.base_model}_{args.dataset}_{args.name}"

    #         eval_metric_dict, ap_list, pp_list = QPP_gpt_result(args)
    #         print(eval_metric_dict)
