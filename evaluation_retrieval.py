import argparse
import os
import pytrec_eval
import json
import pickle
import glob


def evaluation_retrieval(retrieval_results_path, qrel_path, save=True):
    mapping = {"ndcg_cut_3": "ndcg@3",
               "ndcg_cut_10": "ndcg@10",
               "ndcg_cut_100": "ndcg@100",
               "ndcg_cut_1000": "ndcg@1000",
               "mrr_5": "mrr@5",
               "mrr_10": "mrr@10",
               "mrr_100": "mrr@100",
               "map_cut_10": "map@10",
               "map_cut_100": "map@100",
               "map_cut_1000": "map@1000",
               "recall_5": "recall@5",
               "recall_100": "recall@100",
               "recall_1000": 'recall@1000'}


    if 'DLHard_valid' in retrieval_results_path:
        with open('./retrieval_results/bm25_DLHard_valid_result_valid', 'rb') as f:
            run = pickle.load(f)
        
        with open('./datasets/TREC/DLHard/qrels_DLHard.jsonl','r') as f:
            qrel_hard = json.load(f)
        with open('./datasets/TREC/DL2019/qrels_DL2019.jsonl','r') as f:
            qrel_19 = json.load(f)
        with open('./datasets/TREC/DL2020/qrels_DL2020.jsonl','r') as f:
            qrel_20 = json.load(f)

        q19 = list(qrel_19.keys())
        q20 = list(qrel_20.keys())
        qhard = list(qrel_hard.keys())

        qhard_diff = (set(q19) | set(q20)) - set(qhard)
        qrel_all = {**qrel_19, **qrel_20}
        qrel = {qid: qrel_all[qid] for qid in qhard_diff if qid in qrel_all}

    else:
        ''' Load run file '''
        with open(retrieval_results_path, 'rb') as f:
            run = pickle.load(f)
        
        ''' load qrel file '''
        with open(qrel_path,'r') as f:
            qrel = json.load(f)
    
    if len(run) != len(qrel):
        qrel = {key: qrel[key] for key in run}
        
    print("len(list(run))",len(list(run)))
    print("len(list(qrel))",len(list(qrel)))

    run_5 = {}
    run_10 = {}
    run_100 = {}

    for qid,did_score in run.items():
        sorted_did_score = [(did, score) for did, score in sorted(did_score.items(), key=lambda item: item[1], reverse=True)]
        run_5[qid]= dict(sorted_did_score[0:5])
        run_10[qid] = dict(sorted_did_score[0:10])
        run_100[qid] = dict(sorted_did_score[0:100])

    evaluator_ndcg = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut_3','ndcg_cut_10','ndcg_cut_100','ndcg_cut_1000'})
    results_ndcg = evaluator_ndcg.evaluate(run)

    results = {}
    for qid, _ in results_ndcg.items():
        results[qid]={}
        for measure, score in results_ndcg[qid].items():
            results[qid][mapping[measure]]=score

    evaluator_general = pytrec_eval.RelevanceEvaluator(qrel, {'map_cut_10','map_cut_100', 'map_cut_1000','recall_5', 'recall_100','recall_1000'})
    results_general = evaluator_general.evaluate(run)

    for qid, _ in results.items():
        for measure, score in results_general[qid].items():
            results[qid][mapping[measure]] = score

    evaluator_rr = pytrec_eval.RelevanceEvaluator(qrel, {'recip_rank'})
    results_rr_5 = evaluator_rr.evaluate(run_5)
    results_rr_10 = evaluator_rr.evaluate(run_10)
    results_rr_100 = evaluator_rr.evaluate(run_100)

    for qid, _ in results.items():
        results[qid][mapping["mrr_5"]] = results_rr_5[qid]['recip_rank']
        results[qid][mapping["mrr_10"]] = results_rr_10[qid]['recip_rank']
        results[qid][mapping["mrr_100"]] = results_rr_100[qid]['recip_rank']


    for measure in mapping.values():
        overall = pytrec_eval.compute_aggregated_measure(measure, [result[measure] for result in results.values()])
        print('{}: {:.4f}'.format(measure, overall))

    if save:
        base_model, dataset = retrieval_results_path.split("/")[-1].split("_")[:2]
        if 'DLHard_valid' in retrieval_results_path:
            dataset = 'DLHard_valid'

        output_path = f'./output/actual_performance/{base_model}_{dataset}_actual_performance.json'
        if not os.path.exists(f"./output/actual_performance/"):
            os.makedirs(f"./output/actual_performance/")
        
        f = open(output_path, 'w')
        f.write(json.dumps(results))
        f.close()

    return results


def evaluation_retrieval_merge(retrieval_results_path, qrel_path):
    
    path_parts = retrieval_results_path.split('/')
    storage_part = '/'.join(path_parts[:-1])
    file_part = path_parts[-1]
    path_to_files = f"{storage_part}/*{file_part}*" 
    file_names = sorted(glob.glob(path_to_files))

    results = {}
    for file_name in file_names:
        print(file_name)
        results_sub = evaluation_retrieval(file_name, qrel_path, save=False)
        results.update(results_sub)
    
    base_model, dataset = retrieval_results_path.split("/")[-1].split("_")[:2]

    output_path = f'./output/actual_performance/{base_model}_{dataset}_actual_performance.json'
    if not os.path.exists(f"./output/actual_performance/"):
        os.makedirs(f"./output/actual_performance/")
    
    print("start save")
    f = open(output_path, 'w')
    f.write(json.dumps(results))
    f.close()

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset", type=bool, default=False)
    parser.add_argument('--base_model_list', nargs='+', type=str)
    parser.add_argument('--dataset_list', nargs='+', type=str)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    if args.train_dataset:
        for base_model in args.base_model_list:
            for dataset in args.dataset_list:
                retrieval_results_path = f'./retrieval_results/{base_model}_{dataset}_result'
                qrel_path = f'./datasets/TREC/{dataset}/qrels_{dataset}.jsonl'
                evaluation_retrieval_merge(retrieval_results_path, qrel_path)
    else:
        for base_model in args.base_model_list:
            for dataset in args.dataset_list:
                retrieval_results_path = f'./retrieval_results/{base_model}_{dataset}_result'
                qrel_path = f'./datasets/TREC/{dataset}/qrels_{dataset}.jsonl'
                evaluation_retrieval(retrieval_results_path, qrel_path)

# python evaluation_retrieval.py --train_dataset True --base_model_list ance --dataset_list msmarcotrain
# python evaluation_retrieval.py --train_dataset True --base_model_list bm25 --dataset_list msmarcotrain