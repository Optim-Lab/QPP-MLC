import argparse
import os
import json
import pandas as pd


def queries_preprocessing(path, dataset):
    raw_file_path = f'{path}/raw_dataset_queries'
    queries_raw = pd.read_csv('{}/{}_queries.tsv'.format(raw_file_path, dataset), delimiter='\t', keep_default_na=False, header=None)

    if dataset == 'msmarcodev':
        qrels_raw = pd.read_table('{}/{}_qrels.trec'.format(raw_file_path, dataset), sep=' ', header=None)
    elif dataset == 'msmarcotrain':
        qrels_raw = pd.read_csv('{}/{}_qrels.tsv'.format(raw_file_path, dataset), delimiter='\t', keep_default_na=False, header=None)
    else:
        qrels_raw = pd.read_table('{}/{}_qrels.txt'.format(raw_file_path, dataset), sep=' ', header=None)

    queries_raw.columns = ['qid', 'queries']
    qrels_raw.columns = ['qid', 'del', 'pid', 'score']
    qrels_raw = qrels_raw[['qid', 'pid', 'score']]
    
    # len(qrels_raw['qid'].unique())

    qrels_dict = {}
    for i in range(qrels_raw.shape[0]):
        qid, pid, score = qrels_raw.iloc[i]
        qid = str(qid)
        pid = str(pid)
        score = int(score)

        if qid in qrels_dict.keys():
            qrels_dict[qid].update({pid : score})
        else:
            qrels_dict[qid] = {pid : score}

    queries_select = queries_raw[queries_raw['qid'].isin(qrels_raw['qid'].unique())].reset_index(drop=True)
    queries_select['qid'] = queries_select['qid'].astype(str)
    queries_dict = dict(zip(queries_select['qid'], queries_select['queries']))

    path_save = '{}/{}'.format(path, dataset)

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    with open('{}/queries_{}.jsonl'.format(path_save, dataset),'w') as f:
        json.dump(queries_dict, f, ensure_ascii=False, indent=4)

    with open('{}/qrels_{}.jsonl'.format(path_save, dataset),'w') as f:
        json.dump(qrels_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_raw', type=str, required=True)
    parser.add_argument('--dataset_list', nargs='+', type=str, required=True)

    args = parser.parse_args()
    
    for dataset in args.dataset_list:
        queries_preprocessing(path = args.path_raw, dataset = dataset)

#%%
# path = './datasets/TREC'
# dataset_list = ['msmarcotrain', 'msmarcodev', 'DL2019', 'DL2020', 'DLHard']