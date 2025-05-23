#%%
import pandas as pd
import numpy as np
import argparse
import easydict
import json
import os
from scipy.stats import pearsonr, spearmanr, kendalltau
import glob


#%%
def evaluation_QPP(ap_path=None, pp_path=None, target_metric="ndcg@3"):

    '''
    actual_performance
    '''
    ap={}
    with open(ap_path, 'r') as r:
        ap_bank = json.loads(r.read())

    if len(ap_bank) > 10000:
        ap_bank = dict(list(ap_bank.items())[:5000])

    for qid in ap_bank.keys():
        ap[qid]=float(ap_bank[qid][target_metric])

    '''
    predicted_performance
    '''
    pp={}
    with open(pp_path, 'r') as r:
        for line in r:
            qid, pp_value = line.rstrip().split()
            pp[qid]=float(pp_value)
    
    ap_list = []
    pp_list = []

    for qid in ap.keys():
        ap_list.append(ap[qid])
        pp_list.append(pp[qid])

    print(f'sanity check for {target_metric}: {round(np.mean(ap_list),3)}')
    print(f"len_ap: {len(ap)}, len_pp: {len(pp)}")
    print(f"ap's first 5 {ap_list[:5]}")
    print(f"pp's first 5 {pp_list[:5]}")
    
    pearson_coefficient, pearson_pvalue = pearsonr(ap_list, pp_list)
    kendall_coefficient, kendall_pvalue = kendalltau(ap_list, pp_list)
    spearman_coefficient, spearman_pvalue = spearmanr(ap_list, pp_list)
    sMARE, sARE_list = sARE(ap_list, pp_list, rankType='average')

    result_dict = {
        "Pearson": round(pearson_coefficient, 3),
        "Kendall": round(kendall_coefficient, 3),
        # "Spearman": round(spearman_coefficient, 3),
        # "sMARE": round(sMARE, 3),
        "len_ap": len(ap),
        "len_pp": len(pp),
        "P_pvalue": pearson_pvalue,
        "K_pvalue": kendall_pvalue,
        # "S_pvalue": spearman_pvalue
        }

    print(result_dict)

    return result_dict


def evaluation_epochs(ap_path=None, pp_path=None, target_metric="ndcg@3", epoch_num=10):
    for epoch in range(1, epoch_num + 1):
        print(f"Evaluation on epoch {epoch} in terms of {target_metric}.")
        result_dict = evaluation_QPP(ap_path=ap_path, pp_path=pp_path + "-" + str(epoch),target_metric=target_metric)
        print(result_dict)

        with open(pp_path + "." + target_metric, 'a+', encoding='utf-8') as w:
            w.write(str(epoch) + ": " + str(result_dict) + os.linesep)


def evaluation_glob(ap_path=None, pattern=None, target_metrics=["ndcg@3", "ndcg@100", "recall@100", "map@100"]):
    for target_metric in target_metrics:
        for pp_path in sorted(glob.glob(pattern)):
            name = pp_path.split("/")[-1]
            dataset = pp_path.split("/")[-1].split(".")[0]
            output_path ="/".join(pattern.split("/")[:-1])
            pattern_name = pattern.split("/")[-1]

            result_dict = evaluation_QPP(ap_path=ap_path, pp_path=pp_path, target_metric=target_metric)

            with open(f"{output_path}/result.{pattern_name}", 'a+', encoding='utf-8') as w:
                name_=f"{name}-{target_metric}:"
                w.write(f"{name_.ljust(75, ' ')} {str(result_dict)}{os.linesep}")


def sARE(ap_list, pp_list, rankType='average'):

    ap_df = pd.Series(ap_list)
    pp_df = pd.Series(pp_list)

    ap_rank = ap_df.rank(method=rankType)
    pp_rank = pp_df.rank(method=rankType)
    sARE_ = np.abs(pp_rank - ap_rank) / pp_rank.shape[0]
    sMARE = sARE_.mean()
    sARE_list = sARE_.tolist()
    
    return sMARE, sARE_list


def QPP_validation(base_model_list, dataset_list):
    k_list = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 300, 500, 1000]
    x_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    name_list = [
        'clarity-score-k',
        'wig-norm-k',
        'nqc-norm-k',
        'smv-norm-k',
        'qf-score-k',
        'uef-nqc-k',
        'sigma_x',
        # 'sigma_max',
        ]

    df_valid = pd.DataFrame()
    df_valid.index = name_list

    for base_model in base_model_list:
        for dataset in ['msmarcotrain', 'msmarcodev', 'DL2019', 'DL2020', 'DLHard']:
            print(f'{base_model}_{dataset}')

            if dataset in ['msmarcotrain', 'msmarcodev']:
                target_metric = 'mrr@10'
            else:
                target_metric = 'ndcg@10'
    
            optim_k = []
            # j = 1
            for j in range(len(name_list)):

                corr_pear = []
                # i = 1
                for i in range(len(k_list)):

                    if name_list[j] == 'sigma_x':
                        if i >= len(x_list):
                            continue

                        name = f'{name_list[j]}{x_list[i]}'
                    else:
                        name = f'{name_list[j]}{k_list[i]}'

                    ap_path = f'./output/actual_performance/{base_model}_{dataset}_actual_performance.json'
                    pp_path = f'./output/predicted_performance/{base_model}_{dataset}_{name}'

                    result = evaluation_QPP(ap_path, pp_path, target_metric=target_metric)
                    corr_pear.append(result['Pearson'])
                
                if name_list[j] == 'sigma_x':
                    optim_k.append(x_list[np.nanargmax(corr_pear)])
                else:
                    optim_k.append(k_list[np.nanargmax(corr_pear)])

            df_valid[f'{base_model}_{dataset}'] = optim_k

    df_valid.to_csv(f'./output/df_valid.csv')
    df_valid = pd.read_csv(f'./output/df_valid.csv')

    metric_list = ['pearson', 'kendalls']
    value_len = len(metric_list)

    if 'sigma_max' not in name_list:
        name_list.append('sigma_max')
    
    result_table = pd.DataFrame(
        index=range(len(name_list)),
        columns=range(len(base_model_list) * len(dataset_list) * 2)
        )
    
    # name = name_list[1]
    # row = 1
    for row, name in enumerate(name_list):
        for col1, base_model in enumerate(base_model_list):
            for col2, dataset in enumerate(dataset_list):
                print('-------------------------------')
                print(base_model + ' ' + dataset)
                
                if dataset == 'DL2019':
                    dataset_valid = 'DL2020'
                elif dataset == 'DL2020':
                    dataset_valid = 'DL2019'
                elif dataset == 'msmarcodev':
                    dataset_valid = 'msmarcotrain'
                elif dataset == 'DLHard':
                    dataset_valid = 'DLHard_valid'
                
                if name != 'sigma_max':
                    k = df_valid[[f'{base_model}_{dataset_valid}']].iloc[row].values[0]
                
                if name != 'sigma_x':
                    k = int(k)

                ap_path = f'./output/actual_performance/{base_model}_{dataset}_actual_performance.json'

                if name == 'sigma_max':
                    pp_path = f'./output/predicted_performance/{base_model}_{dataset}_{name}'
                else:
                    pp_path = f'./output/predicted_performance/{base_model}_{dataset}_{name}{k}'
                            
                if dataset in ['msmarcotrain', 'msmarcodev']:
                    result_dict = evaluation_QPP(ap_path=ap_path, pp_path=pp_path, target_metric='mrr@10')
                else:
                    result_dict = evaluation_QPP(ap_path=ap_path, pp_path=pp_path, target_metric='ndcg@10')
                    
                print('-------------------------------')

                result_value = [result_dict['Pearson'], result_dict['Kendall']]

                result_table.iloc[row, (col1*len(dataset_list)*value_len + col2*value_len):(col1*len(dataset_list)*value_len + ((col2+1)*value_len))] = result_value

    combined_list = [(base_model, dataset, metric) 
                    for base_model in base_model_list
                    for dataset in dataset_list
                    for metric in metric_list]
    combined_list = pd.DataFrame(combined_list)
    multi_columns = pd.MultiIndex.from_arrays([combined_list[0], combined_list[1], combined_list[2]], names=('model', 'dataset', 'metric'))

    result_table.index = name_list
    result_table.columns = multi_columns

    save_path = './output/QPP_results_table.csv'
    result_table.to_csv(save_path)

    return df_valid


#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--ap_path", type=str)
    parser.add_argument("--pp_path", type=str)
    parser.add_argument("--target_metric", type=str)
    parser.add_argument("--pattern", type=str, default=None)
    parser.add_argument("--epoch_num", type=int, default=None)

    parser.add_argument("--base_model_list", nargs='+', type=str, default=[])
    parser.add_argument("--dataset_list", nargs='+', type=str, default=[])
    
    try:
        args = parser.parse_args()
    except:
        args = easydict.EasyDict({
            'ap_path' : 1,
            'pp_path' : 1,
            'pattern' : None,
            'epoch_num' : None,
            'target_metric' : 'mrr@10',
        })
        base_model = 'ance'
        dataset = 'DL2020'
        name = 'wig-no-norm-5'
        args.ap_path = f'./output/actual_performance/{base_model}_{dataset}_actual_performance.json'
        args.pp_path = f'./output/predicted_performance/{base_model}_{dataset}_{name}'

        args.base_model_list = ['bm25', 'ance']
        args.dataset_list = ['msmarcodev', 'DL2019', 'DL2020', 'DLHard']

    # if args.epoch_num is not None:
    #     result = evaluation_epochs(args.ap_path, args.pp_path, target_metric=args.target_metric, epoch_num=args.epoch_num)
    # elif args.pattern is not None:
    #     evaluation_glob(args.ap_path, args.pattern, target_metrics=args.target_metrics)
    # else:
    #     result = evaluation_QPP(args.ap_path, args.pp_path, target_metric=args.target_metric)

    # args.base_model_list = ['bm25', 'ance']
    # args.dataset_list = ['msmarcodev', 'DL2019', 'DL2020', 'DLHard']

    df_valid = QPP_validation(args.base_model_list, args.dataset_list)

# python evaluation_QPP.py \
#   --base_model_list bm25 ance \
#   --dataset_list msmarcodev DL2019 DL2020 DLHard