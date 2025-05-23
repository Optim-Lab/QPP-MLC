#%%
import os
import sys
import copy
import glob
import random
import torch
import argparse
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, kendalltau, gaussian_kde, norm, chi2_contingency, ranksums
from scipy.optimize import brentq
pd.options.display.float_format = '{:.3f}'.format
torch.set_printoptions(precision=3, sci_mode=False)

os.chdir('/root/default/qppmlc')
sys.path.append(os.getcwd())
print(os.getcwd())

from model import QPP_MLC
from driver import Driver
from dataset import DatasetQPPCross, collate_fn_cross
from utils import set_random_seed, str2bool

#%%

def training_QPP_MLC(args):

    weight_dir = f"{args.checkpoint_path}weight_{args.setup}/"
    weight_path_list = sorted(glob.glob(f"{weight_dir}weight_*{args.setup}*"))

    model = QPP_MLC(args)

    ###
    # params_model = sum(p.numel() for p in model.parameters())
    # params_bert = sum(p.numel() for p in model.bert.parameters())
    # params_transformer = sum(p.numel() for p in model.transformer.parameters())
    # params_mlp = sum(p.numel() for p in model.mlp.parameters())
    ###

    if args.mode == 'pilot':
        for param in model.bert.parameters():
            param.requires_grad = False
    
    if not len(weight_path_list) == 0:
        model.load_state_dict(torch.load(weight_path_list[-1]))
        print(f"{args.name} model load!!")
        print(f"path : {weight_path_list[-1]}")
        start_epoch = int(weight_path_list[-1][-9:-7])
        start_split = int(weight_path_list[-1][-6:-4])
    else:
        start_epoch = 1
        start_split = 1

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    driver = Driver(args, model)
    optimizer.zero_grad()
    
    if args.mode == 'pilot':
        num_split_last = 10
    else:
        num_split_last = args.num_split
    
    split_id = 1
    epoch_id = 1
    fold_id = 1
    for epoch_id in range(start_epoch, args.epoch_num + 1):
        print(f'epoch: {epoch_id}')

        if args.bert_epoch and epoch_id != 1:
            for param in model.bert.parameters():
                param.requires_grad = False
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_per = trainable_params / total_params
        print(f"total_params: {total_params}, trainable_params: {trainable_params}, trainable_per: {round(trainable_per, 5)}")
        
        for split_id in range(start_split, num_split_last + 1):
            dataset = DatasetQPPCross(args, split_id=split_id)
            result_dict, result_dict_tilde = driver.training(dataset, collate_fn_cross, epoch_id, split_id, fold_id, optimizer)
            
    return result_dict


def concept_shift(args):
    transform = 'logit'
    top_j = 0
    base_model = 'bm25'
    base_model = 'ance'

    # dataset_name = "msmarcotrain"
    # dataset_name = "msmarcodev"
    # dataset_name = "DL2020"
    dataset_list = ['msmarcotrain', 'msmarcodev', 'DL2019', 'DL2020', 'DLHard']
    df = pd.DataFrame()
    for dataset_name in dataset_list:
        dict_embed, args_infer = load_QPP_output(args, dataset_name, base_model = base_model, split_id=1)

        pr = dict_embed['predicted_relevance'].cpu()
        if args_infer.dataset in ['DL2019', 'DL2020', 'DLHard', 'DLHard_valid']:
            ar = (dict_embed['actual_relevance'] >= 2).int().cpu()
        else:
            ar = dict_embed['actual_relevance'].cpu()
        
        pr_array = np.array(pr[:, top_j])
        ar_array = np.array(ar[:, top_j])
        if transform == 'log':
            pr_array = -np.log(1 - pr_array)
        elif transform == 'logit':
            # logit(pr_array)
            pr_array = np.log(pr_array / (1-pr_array))

        if dataset_name == 'msmarcotrain':
            set_random_seed(seed=args.random_seed)
            
            all_indices = list(range(pr_array.shape[0]))
            random.shuffle(all_indices)

            pr_array = pr_array[all_indices[:5000]]
            ar_array = ar_array[all_indices[:5000]]

        
        pr_0 = pr_array[ar_array == 0]
        pr_1 = pr_array[ar_array == 1]
        prop_rel = len(pr_1) / len(pr_array)

        if dataset_name == 'msmarcotrain':
            pr_0_train = pr_0
            pr_1_train = pr_1
            pr_array_train = pr_array
            ar_array_train = ar_array

        # wilcoxon rank-sum test, chi-square test
        wilcoxon_stat, pval_wilcoxon = wilcoxon_rank_sum_test(pr_array, pr_array_train)
        wilcoxon_stat_0, pval_wilcoxon_0 = wilcoxon_rank_sum_test(pr_0, pr_0_train)
        wilcoxon_stat_1, pval_wilcoxon_1 = wilcoxon_rank_sum_test(pr_1, pr_1_train)
        chi, chi_pval = prop_chi2_test(ar_array, ar_array_train)

        pi_0 = 1 - prop_rel
        pi_1 = prop_rel
        mu_0 = pr_0.mean()
        mu_1 = pr_1.mean()
        sigma_0 = pr_0.std()
        sigma_1 = pr_1.std()
        mu = pr_array.mean()
        sigma = pr_array.std()

        df_ = pd.DataFrame((
            [pi_0, chi_pval, mu, sigma**2, pval_wilcoxon, mu_0, sigma_0**2, pval_wilcoxon_0],
            [pi_1, chi_pval, mu, sigma**2, pval_wilcoxon, mu_1, sigma_1**2, pval_wilcoxon_1],
            )).T
        df = pd.concat([df, df_], axis=1)

    df.index = ['pi', 'pval_chi', 'mu', 'sigma_squared', 'pval_wilcoxon', 'mu', 'sigma_squared', 'pval_wilcoxon']
    df.columns = [f"{name}_R={r}" for name in dataset_list for r in [0, 1]]
    df

    return df


def wilcoxon_rank_sum_test(pr_0, pr_1):
    stat, pval = ranksums(pr_0, pr_1)

    return stat, pval


def prop_chi2_test(ar_array_train, ar_array):
    x1 = np.sum(ar_array_train)              # R=1 in train
    x2 = np.sum(ar_array)               # R=1 in test
    n1 = len(ar_array_train)
    n2 = len(ar_array)

    contingency_table = [
        [x1, n1 - x1],
        [x2, n2 - x2]
    ]

    # chi-squared test
    chi2, pval, dof, expected = chi2_contingency(contingency_table)
    if (expected <= 5).sum():
        print('cell expected lower then 5')

    return chi2, pval


def find_f1_threshold(pr_0, pr_1, prop_rel, gaussian=True, x_range=None, resolution=500):

    # ance, DL2020
    # mu_0, std_0 = (-2.2184901, 1.4210249)
    # mu_1, std_1 = (-0.39660847, 1.4222918)

    # bm25, dev
    # prop_rel = 0.067
    # mu_0, std_0 = (-5.347016, 2.7492025)
    # mu_1, std_1 = (-1.5405025, 1.3477211)

    # Gaussian 정규분포 추정
    mu_0, std_0 = np.mean(pr_0), np.std(pr_0)
    mu_1, std_1 = np.mean(pr_1), np.std(pr_1)

    # x 축 범위
    if x_range is None:
        # x_min = -20
        # x_max = 20
        x_min = min(mu_0 - 4 * std_0, mu_1 - 4 * std_1)
        x_max = max(mu_0 + 4 * std_0, mu_1 + 4 * std_1)
    else:
        x_min, x_max = x_range

    x_vals = np.linspace(x_min, x_max, resolution)

    if gaussian:
        # 정규분포 PDF 계산
        pdf_0 = (1 - prop_rel) * norm.pdf(x_vals, loc=mu_0, scale=std_0)
        pdf_1 = prop_rel * norm.pdf(x_vals, loc=mu_1, scale=std_1)

        diff = pdf_1 / pdf_0 - ((prop_rel * (1 - norm.cdf(x_vals, mu_1, std_1))) / (prop_rel + (1-prop_rel) * (1-norm.cdf(x_vals, mu_0, std_0))))
        
        crossings = np.where(np.diff(np.sign(diff)))[0]
        thresholds = []

        for i in crossings:
            try:
                t = brentq(lambda x:
                            (prop_rel * norm.pdf(x, mu_1, std_1)) / ((1 - prop_rel) * norm.pdf(x, mu_0, std_0)) -
                            ((prop_rel * (1 - norm.cdf(x, mu_1, std_1))) / (prop_rel + (1-prop_rel) * (1-norm.cdf(x, mu_0, std_0)))), x_vals[i], x_vals[i + 1])

                thresholds.append(t)

            except ValueError:
                continue

        thresholds.append(-float("inf"))
        thresholds.append(float("inf"))
        thresholds.sort()
    
        TP = prop_rel * (1 - norm.cdf(thresholds, loc=mu_1, scale=std_1))
        TN = (1 - prop_rel) * norm.cdf(thresholds, loc=mu_0, scale=std_0)
        FP = (1 - prop_rel) * (1 - norm.cdf(thresholds, loc=mu_0, scale=std_0))
        FN = prop_rel * norm.cdf(thresholds, loc=mu_1, scale=std_1)
    
    else:
        # KDE 계산
        kde_0 = gaussian_kde(pr_0)
        kde_1 = gaussian_kde(pr_1)

        pdf_0 = (1 - prop_rel) * kde_0(x_vals)
        pdf_1 = prop_rel * kde_1(x_vals)
        
        diff = pdf_1 / pdf_0 - ((prop_rel * (1 - np.array([kde_1.integrate_box_1d(-np.inf, x) for x in x_vals]))) / (prop_rel + (1-prop_rel) * (1-np.array([kde_0.integrate_box_1d(-np.inf, x) for x in x_vals]))))

        crossings = np.where(np.diff(np.sign(diff)))[0]
        thresholds = []

        for i in crossings:
            try:
                t = brentq(lambda x:
                            (prop_rel * kde_1(x)) / ((1 - prop_rel) * kde_0(x)) -
                            ((prop_rel * (1 - kde_1.integrate_box_1d(-np.inf, x))) / (prop_rel + (1-prop_rel) * (1-kde_0.integrate_box_1d(-np.inf, x)))), x_vals[i], x_vals[i + 1])
                
                thresholds.append(t)
                    
            except ValueError:
                continue
    
        thresholds.append(-float("inf"))
        thresholds.append(float("inf"))
        thresholds.sort()
    
        cdf_0 = np.array([kde_0.integrate_box_1d(-np.inf, thres) for thres in thresholds])
        cdf_1 = np.array([kde_1.integrate_box_1d(-np.inf, thres) for thres in thresholds])

        TP = prop_rel * (1 - cdf_1)
        TN = (1 - prop_rel) * cdf_0
        FP = (1 - prop_rel) * (1 - cdf_0)
        FN = prop_rel * cdf_1
    

    acc = (TP + TN) / (TP + TN + FP +FN)
    error = 1 - (TP + TN) / (TP + TN + FP +FN)
    sens = TP / (TP +FN)
    prec = TP / (TP +FP)
    f1 = 2 / (1 / sens + 1 / prec)
    metric = [acc, error, sens, prec, f1]

    best_thresh = thresholds[np.nanargmin(-f1)]

    return best_thresh, x_vals, pdf_0, pdf_1, thresholds, metric


def find_error_threshold(pr_0, pr_1, prop_rel, gaussian=True, x_range=None, resolution=500):
    # ex
    # pr_0 = np.array([0.231, 0.254, 0.123, 0.632])
    # pr_1 = np.array([0.221, 0.514, 0.323, 0.132, 0.323, 0.654])

    # ance, DL2020
    # mu_0, std_0 = (-2.2184901, 1.4210249)
    # mu_1, std_1 = (-0.39660847, 1.4222918)

    # bm25, dev
    # mu_0, std_0 = (-5.347016, 2.7492025)
    # mu_1, std_1 = (-1.5405025, 1.3477211)

    # Gaussian 정규분포 추정
    mu_0, std_0 = np.mean(pr_0), np.std(pr_0)
    mu_1, std_1 = np.mean(pr_1), np.std(pr_1)

    # x 축 범위
    if x_range is None:
        x_min = min(mu_0 - 4 * std_0, mu_1 - 4 * std_1)
        x_max = max(mu_0 + 4 * std_0, mu_1 + 4 * std_1)
    else:
        x_min, x_max = x_range

    # x_vals = np.linspace(-50, 50, 5000)
    x_vals = np.linspace(x_min, x_max, resolution)

    thresholds = []

    if gaussian:
        # 정규분포 PDF 계산
        pdf_0 = (1 - prop_rel) * norm.pdf(x_vals, loc=mu_0, scale=std_0)
        pdf_1 = prop_rel * norm.pdf(x_vals, loc=mu_1, scale=std_1)
        
        # 로그 prior + 스케일링
        delta = np.log(((1-prop_rel) * std_1) / (prop_rel * std_0))

        # 계수
        A = 1 / (2 * std_0**2) - 1 / (2 * std_1**2)
        B = -(mu_0 / std_0**2 - mu_1 / std_1**2)
        C = (mu_0**2) / (2 * std_0**2) - (mu_1**2) / (2 * std_1**2) - delta

        # 판별식
        D = B**2 - 4 * A * C
        sqrt_D = np.sqrt(D)
        tau_1 = (-B - sqrt_D) / (2 * A)
        tau_2 = (-B + sqrt_D) / (2 * A)

        # thresholds
        thresholds.append(-float("inf"))
        thresholds.append(tau_1)
        thresholds.append(tau_2)
        thresholds.append(float("inf"))
        thresholds.sort()

        TP = prop_rel * (1 - norm.cdf(thresholds, loc=mu_1, scale=std_1))
        TN = (1 - prop_rel) * norm.cdf(thresholds, loc=mu_0, scale=std_0)
        FP = (1 - prop_rel) * (1 - norm.cdf(thresholds, loc=mu_0, scale=std_0))
        FN = prop_rel * norm.cdf(thresholds, loc=mu_1, scale=std_1)
    
    else:
        # KDE 계산
        kde_0 = gaussian_kde(pr_0)
        kde_1 = gaussian_kde(pr_1)

        pdf_0 = (1 - prop_rel) * kde_0(x_vals)
        pdf_1 = prop_rel * kde_1(x_vals)

        # solution searching
        diff = pdf_1 - pdf_0
        crossings = np.where(np.diff(np.sign(diff)))[0]

        for i in crossings:
            try:
                t = brentq(lambda x: prop_rel * kde_1(x) - (1 - prop_rel) * kde_0(x),
                        x_vals[i], x_vals[i + 1])
            
                # eps=0.01 보다 큰 pdf 값을 갖는 교점만 threshold로 채택
                if prop_rel * kde_1(t) > 0.01:
                    thresholds.append(t)

            except ValueError:
                continue
        
        thresholds.append(-float("inf"))
        thresholds.append(float("inf"))
        thresholds.sort()
    
        cdf_0 = np.array([kde_0.integrate_box_1d(-np.inf, thres) for thres in thresholds])
        cdf_1 = np.array([kde_1.integrate_box_1d(-np.inf, thres) for thres in thresholds])

        TP = prop_rel * (1 - cdf_1)
        TN = (1 - prop_rel) * cdf_0
        FP = (1 - prop_rel) * (1 - cdf_0)
        FN = prop_rel * cdf_1
    

    acc = (TP + TN) / (TP + TN + FP +FN)
    error = 1 - (TP + TN) / (TP + TN + FP +FN)
    sens = TP / (TP + FN)
    prec = TP / (TP + FP)
    f1 = 2 / (1 / sens + 1 / prec)
    metric = [acc, error, sens, prec, f1]

    best_thresh = thresholds[np.nanargmin(error)]
    
    return best_thresh, x_vals, pdf_0, pdf_1, thresholds, metric


def bayes_threshold(pr, ar, transform='logit', loss='error', gaussian=True, pool=0):

    # calculate bayes threshold
    # pool=5
    # gaussian=False
    # top_j = 0
    # top_j = 5
    best_thresh_list = []
    for top_j in range(10):
        # print(f'top_{top_j+1} 번째 문서.')
        if top_j >= 10 - pool:
            pr_array = np.array(pr[:, top_j:]).reshape(-1)
            ar_array = np.array(ar[:, top_j:]).reshape(-1)
        else:
            pr_array = np.array(pr[:, top_j])
            ar_array = np.array(ar[:, top_j])

        if transform == 'log':
            pr_array = -np.log(1 - pr_array)
        elif transform == 'logit':
            # logit(pr_array)
            pr_array = np.log(pr_array / (1-pr_array))

        pr_0 = pr_array[ar_array == 0]
        pr_1 = pr_array[ar_array == 1]
        prop_rel = len(pr_1) / len(pr_array)

        if len(pr_0) < 2:
            best_thresh = -9
            best_thresh_list.append(best_thresh)
            continue
        elif len(pr_1) < 2:
            best_thresh = 9
            best_thresh_list.append(best_thresh)
            continue
        
        if loss == 'error':
            best_thresh, x_vals, pdf_0, pdf_1, thresholds, metric = find_error_threshold(pr_0, pr_1, prop_rel, gaussian=gaussian)
        elif loss == 'f1':
            best_thresh, x_vals, pdf_0, pdf_1, thresholds, metric = find_f1_threshold(pr_0, pr_1, prop_rel, gaussian=gaussian)

        if top_j >= 10 - pool:
            best_thresh_list.extend([best_thresh] * pool)
            break
        else:
            best_thresh_list.append(best_thresh)
    
    bayes_threshold_array = 1 / (1+(np.exp((-np.array(best_thresh_list)))))

    return bayes_threshold_array


def self_validation(args, dataset_name, base_model, prop_valid, num_iter=100, loss='error', gaussian=True, pool=7, select_pool=False):
    '''
    - DL2019, DL2020, DLHard 에서 일정 비율(a%)를 validation dataset으로 활용하여 각 데이터셋에서 threshold를 정한다.
    - threshold를 정하는 방법
        - top_k개 각 레이블마다 특정 metric을 최대로 하는 방법 (Scut)
        - youden_thres (metric: AUC)
        - f1_thres (metric: f1-score)
        - mean_thres
        - median_thres
        - quantile_thres

    '''

    ''' valid dataset 생성 '''
    set_random_seed(seed=args.random_seed)

    split_id = 1
    # base_model = 'ance'
    # base_model = 'bm25'
    # dataset_name = "msmarcotrain"
    # dataset_name = "msmarcodev"
    # dataset_name = "DL2019"
    # dataset_name = "DL2020"
    # dataset_name = "DLHard"
    # dataset_name = "DLHard_valid"
    
    dict_embed, args_infer = load_QPP_output(args, dataset_name, base_model = base_model, split_id=split_id)
    dict_embed.keys()
    args_infer.err = False
    
    # num_iter = 3
    # num_iter = 100
    # i = 0
    # prop_valid = 0.2
    coef_dcg_pear_list = []
    coef_dcg_kend_list = []
    coef_ndcg_pear_list = []
    coef_ndcg_kend_list = []
    for i in range(num_iter):

        # 전체 sample index 에서 prop_valid% 만큼 랜덤 샘플링
        num_q = len(dict_embed['qid_list'])
        num_valid = round(num_q * prop_valid)
        # num_test = num_q - num_valid

        all_indices = list(range(num_q))
        random.shuffle(all_indices)
        valid_indices = all_indices[:num_valid]  # 10개 valid
        test_indices = all_indices[num_valid:]   # 나머지 33개 test

        # dict_embed를 valid와 test로 분할
        dict_embed_valid = {key: value[valid_indices] for key, value in dict_embed.items()}
        dict_embed_test = {key: value[test_indices] for key, value in dict_embed.items()}

        ''' valid dataset을 이용하여 threshold 계산 '''
        if args_infer.dataset in ['DL2019', 'DL2020', 'DLHard', 'DLHard_valid']:
            ar = (dict_embed_valid['actual_relevance'] >= 2).int().cpu()
        else:
            ar = dict_embed_valid['actual_relevance'].cpu()
        
        pr = dict_embed_valid['predicted_relevance'].cpu()

        # calculate youden and f1 thres 
        # youden_threshold_array, f1_threshold_array = youden_f1_thres(pr, ar, roc_curve_bool=False)

        # calculate bayes threshold
        # if dataset_name in ['msmarcotrain', 'msmarcodev']:
        #     gaussian = False
        # else:
        #     gaussian = True

        # select_pool = True
        if select_pool:
            if loss == 'error':
                error_threshold_array_1 = bayes_threshold(pr, ar, transform='logit', loss=loss, gaussian=gaussian, pool=1)
                error_threshold_array_10 = bayes_threshold(pr, ar, transform='logit', loss=loss, gaussian=gaussian, pool=10)
                threshold_pool_1 = torch.tensor(error_threshold_array_1).cuda()
                threshold_pool_10 = torch.tensor(error_threshold_array_10).cuda()

            elif loss == 'f1':
                f1_threshold_array_1 = bayes_threshold(pr, ar, transform='logit', loss=loss, gaussian=gaussian, pool=1)
                f1_threshold_array_10 = bayes_threshold(pr, ar, transform='logit', loss=loss, gaussian=gaussian, pool=10)
                threshold_pool_1 = torch.tensor(f1_threshold_array_1).cuda()
                threshold_pool_10 = torch.tensor(f1_threshold_array_10).cuda()
            
            dcg_ndcg_pear_coef_1, _, _, _ = calculate_corr(dict_embed_valid, threshold=threshold_pool_1, top_k=args_infer.top_k, target_metric=args_infer.target_metric, err_bool=False)
            dcg_ndcg_pear_coef_10, _, _, _ = calculate_corr(dict_embed_valid, threshold=threshold_pool_10, top_k=args_infer.top_k, target_metric=args_infer.target_metric, err_bool=False)

            if np.isnan(dcg_ndcg_pear_coef_1):
                dcg_ndcg_pear_coef_1 = -1
            if np.isnan(dcg_ndcg_pear_coef_10):
                dcg_ndcg_pear_coef_10 = -1

            if dcg_ndcg_pear_coef_1 > dcg_ndcg_pear_coef_10:
                dcg_ndcg_pear_coef, dcg_ndcg_kend_coef, ndcg_ndcg_pear_coef, ndcg_ndcg_kend_coef = calculate_corr(dict_embed_test, threshold=threshold_pool_1, top_k=args_infer.top_k, target_metric=args_infer.target_metric, err_bool=False)
            else:
                dcg_ndcg_pear_coef, dcg_ndcg_kend_coef, ndcg_ndcg_pear_coef, ndcg_ndcg_kend_coef = calculate_corr(dict_embed_test, threshold=threshold_pool_10, top_k=args_infer.top_k, target_metric=args_infer.target_metric, err_bool=False)

            coef_dcg_pear_list.append(dcg_ndcg_pear_coef)
            coef_dcg_kend_list.append(dcg_ndcg_kend_coef)
            coef_ndcg_pear_list.append(ndcg_ndcg_pear_coef)
            coef_ndcg_kend_list.append(ndcg_ndcg_kend_coef)

        else:
            
            if loss == 'error':
                error_threshold_array = bayes_threshold(pr, ar, transform='logit', loss=loss, gaussian=gaussian, pool=pool)
                args_infer.threshold = torch.tensor(error_threshold_array).cuda()
            elif loss == 'f1':
                f1_threshold_array = bayes_threshold(pr, ar, transform='logit', loss=loss, gaussian=gaussian, pool=pool)
                args_infer.threshold = torch.tensor(f1_threshold_array).cuda()
            
            ''' test dataset에서 성능 확인 '''
            dcg_ndcg_pear_coef, dcg_ndcg_kend_coef, ndcg_ndcg_pear_coef, ndcg_ndcg_kend_coef = calculate_corr(dict_embed_test, threshold=args_infer.threshold, top_k=args_infer.top_k, target_metric=args_infer.target_metric, err_bool=False)

            coef_dcg_pear_list.append(dcg_ndcg_pear_coef)
            coef_dcg_kend_list.append(dcg_ndcg_kend_coef)
            coef_ndcg_pear_list.append(ndcg_ndcg_pear_coef)
            coef_ndcg_kend_list.append(ndcg_ndcg_kend_coef)
        

    return coef_dcg_pear_list, coef_dcg_kend_list, coef_ndcg_pear_list, coef_ndcg_kend_list

# base_model = 'ance'
# # # base_model = 'bm25'
# dataset_name = 'DL2019'
# # dataset_name = 'DL2020'
# # dataset_name = 'DLHard'
# # # thres_type = 'youden'
# # prop_valid = 0.5
# num_iter = 10
# prop_valid_range = np.arange(0.1, 1.00, 0.1)
# thres_type = 'bayes_gauss'
# dcg_ndcg = 'dcg'


def self_validation_result(args, prop_valid_range, num_iter, dcg_ndcg='dcg', loss='error', gaussian=True, pool=7, save_df=False, select_pool=False, valid_method='self'):

    base_model_list = ['bm25', 'ance']
    if valid_method == 'loocv':
        dataset_list = ['DL2019', 'DL2020', 'DLHard']
        prop_valid_range = np.array([0.01])
    else:
        dataset_list = ['msmarcodev', 'DL2019', 'DL2020', 'DLHard']
    
    ###
    # dataset_list = ['DL2019', 'DL2020', 'DLHard']
    ###

    df_corr_mean = pd.DataFrame()
    df_corr_std = pd.DataFrame()
    for base_model in base_model_list:
        corr_mean_list = []
        corr_std_list = []

        for dataset_name in dataset_list:
            corr_mean_pear_list = []
            corr_mean_kend_list = []
            corr_std_pear_list = []
            corr_std_kend_list = []

            for prop_valid in prop_valid_range:
                # print(f'{base_model}, {dataset_name}, {prop_valid}')

                if valid_method == 'self':
                    coef_dcg_pear_list, coef_dcg_kend_list, coef_ndcg_pear_list, coef_ndcg_kend_list = self_validation(args, dataset_name, base_model, prop_valid, num_iter=num_iter, loss=loss, gaussian=gaussian, pool=pool, select_pool=select_pool)
                # elif valid_method == 'loocv':
                #     coef_dcg_pear_list, coef_dcg_kend_list, coef_ndcg_pear_list, coef_ndcg_kend_list = self_LOOCV(args, dataset_name, base_model, loss=loss, gaussian=gaussian, pool=pool, select_pool=select_pool)
                # elif valid_method == 'gpt':
                #     coef_dcg_pear_list, coef_dcg_kend_list, coef_ndcg_pear_list, coef_ndcg_kend_list = self_gpt(args, dataset_name, base_model, loss=loss, gaussian=gaussian, pool=pool, select_pool=select_pool)
                # elif valid_method == 'train':
                #     coef_dcg_pear_list, coef_dcg_kend_list, coef_ndcg_pear_list, coef_ndcg_kend_list = self_train(args, dataset_name, base_model, loss=loss, gaussian=gaussian, pool=pool, select_pool=select_pool)
                
                if dcg_ndcg == 'dcg':
                    coef_pear_list = coef_dcg_pear_list
                    coef_kend_list = coef_dcg_kend_list
                elif dcg_ndcg == 'ndcg':
                    coef_pear_list = coef_ndcg_pear_list
                    coef_kend_list = coef_ndcg_kend_list


                coef_pear_array = np.array(coef_pear_list)
                coef_kend_array = np.array(coef_kend_list)
                print(f'nan 비율: {np.isnan(coef_pear_array).mean()}')

                corr_mean_pear_list.append(np.nanmean(coef_pear_array))
                corr_mean_kend_list.append(np.nanmean(coef_kend_array))
                corr_std_pear_list.append(np.nanstd(coef_pear_array))
                corr_std_kend_list.append(np.nanstd(coef_kend_array))

            corr_mean_list.append(corr_mean_pear_list)
            corr_mean_list.append(corr_mean_kend_list)
            corr_std_list.append(corr_std_pear_list)
            corr_std_list.append(corr_std_kend_list)

        df_corr_mean_ = pd.DataFrame(corr_mean_list).T
        df_corr_std_ = pd.DataFrame(corr_std_list).T
        df_corr_mean = pd.concat([df_corr_mean, df_corr_mean_], axis=1)
        df_corr_std = pd.concat([df_corr_std, df_corr_std_], axis=1)
    
    columns = [f"{dataset}_{suffix}" for dataset in dataset_list for suffix in ['pear', 'kend']] * 2
    index = [f"{loss}_{round(prop_valid, 2)}" for prop_valid in prop_valid_range]
    index_std = [f"std_{loss}_{round(prop_valid, 2)}" for prop_valid in prop_valid_range]

    df_corr_mean.columns = columns
    df_corr_mean.index = index
    df_corr_std.columns = columns
    df_corr_std.index = index_std

    if save_df:
        df_result_self_valid = pd.concat([df_corr_mean, df_corr_std])
        if select_pool:
            if valid_method=='loocv':
                df_result_self_valid.to_csv(f'/root/default/Neural-IR/output/df_result_self_loocv_{dcg_ndcg}_{loss}_{gaussian}_select.csv')
            elif valid_method=='self':
                df_result_self_valid.to_csv(f'/root/default/Neural-IR/output/df_result_self_valid_{dcg_ndcg}_{loss}_{gaussian}_select.csv')
            elif valid_method=='gpt':
                df_result_self_valid.to_csv(f'/root/default/Neural-IR/output/df_result_self_gpt_{dcg_ndcg}_{loss}_{gaussian}_select.csv')
            elif valid_method=='train':
                df_result_self_valid.to_csv(f'/root/default/Neural-IR/output/df_result_self_train_{dcg_ndcg}_{loss}_{gaussian}_select.csv')
        else:
            if valid_method=='loocv':
                df_result_self_valid.to_csv(f'/root/default/Neural-IR/output/df_result_self_loocv_{dcg_ndcg}_{loss}_{gaussian}_{pool}.csv')
            elif valid_method=='self':
                df_result_self_valid.to_csv(f'/root/default/Neural-IR/output/df_result_self_valid_{dcg_ndcg}_{loss}_{gaussian}_{pool}.csv')
            elif valid_method=='gpt':
                df_result_self_valid.to_csv(f'/root/default/Neural-IR/output/df_result_self_gpt_{dcg_ndcg}_{loss}_{gaussian}_{pool}.csv')
            elif valid_method=='train':
                df_result_self_valid.to_csv(f'/root/default/Neural-IR/output/df_result_self_train_{dcg_ndcg}_{loss}_{gaussian}_{pool}.csv')
            
    return df_corr_mean, df_corr_std

# import warnings
# warnings.filterwarnings("ignore")

# num_iter = 100
# prop_valid_range = np.array([0.01])
# prop_valid_range = np.arange(0.1, 1.0, 0.1)
# prop_valid_range = np.array([0.3])
# dcg_ndcg = 'dcg'
# save_df=True
# select_pool=True
# valid_method = 'train' # 'loocv', 'self', 'gpt', 'train'

# for loss in ['f1']:
#     for gaussian in [True]:
#         for pool in [1]:
#             print(f'{loss},{gaussian},{pool}')
#             df_corr_mean, df_corr_std = self_validation_result(
#                 args,
#                 prop_valid_range,
#                 num_iter,
#                 dcg_ndcg=dcg_ndcg,
#                 loss=loss,
#                 gaussian=gaussian,
#                 pool=pool,
#                 save_df=save_df,
#                 select_pool=select_pool,
#                 valid_method=valid_method
#                 )

#             print(f'loss: {loss}, gaussian: {gaussian}, pool: {pool}')
#             print(df_corr_mean)


def load_QPP_output(args, dataset_name, base_model, split_id=1, load_dict=True):

    args.base_model = base_model
    args_infer = copy.deepcopy(args)
    args_infer.dataset = dataset_name
    if dataset_name in ['DL2019', 'DL2020', 'DLHard', 'DLHard_valid']:
        args_infer.target_metric = 'ndcg@10'
    else:
        args_infer.target_metric = 'mrr@10'

    args_infer.qrels_path = f'./datasets/TREC/{args_infer.dataset}/qrels_{args_infer.dataset}.jsonl'
    args_infer.query_path = f'./datasets/TREC/{args_infer.dataset}/queries_{args_infer.dataset}.jsonl'
    args_infer.ap_path = f'./output/actual_performance/{args_infer.base_model}_{args_infer.dataset}_actual_performance.json'
    args_infer.setup = f"{args_infer.base_model}_{args_infer.dataset}_{args_infer.name}_{args_infer.target_metric}"
    args_infer.setup_dataset = f"{args_infer.base_model}_{args_infer.dataset}_{args_infer.target_metric}"
    args_infer.batch_size = int(32)
    
    if load_dict:
        # data load (R_hat)
        data_dir = f"{args_infer.checkpoint_path}data_{args_infer.setup}/"
        data_path = f"{data_dir}data_{args_infer.setup_dataset}_{split_id:02}.pkl"
        dict_embed = torch.load(data_path)
        # dict_embed.keys()
        return dict_embed, args_infer
    
    else:
        return args_infer

def calculate_corr(dict_embed, threshold, top_k=10, target_metric='ndcg@10', err_bool=False, corr_bool=True):
    
    # threshold = args_infer.threshold
    if not err_bool:
        if len(threshold.shape) == 2:
            predicted_relevance = torch.where(dict_embed['predicted_relevance'] < threshold[0], 0, 
                                    torch.where(dict_embed['predicted_relevance'] < threshold[1], 1,
                                    torch.where(dict_embed['predicted_relevance'] < threshold[2], 2, 3)))
        else:
            predicted_relevance = (dict_embed['predicted_relevance'] > threshold).int()
    else:
        predicted_relevance = dict_embed['predicted_relevance']

    ''' Compute err '''
    predicted_relevance_partial = predicted_relevance[:,:top_k]
    batch_size, top_m = predicted_relevance_partial.shape
    
    # Compute cumulative product of (1 - r_j) for ERR weight calculation
    one_minus_relevance = 1 - predicted_relevance_partial
    prod_cum = torch.cumprod(one_minus_relevance, dim=1)
    
    # Adjust cumulative product so that prod_cum[:, i] represents \prod_{j=1}^{i-1} (1 - r_j)
    prod_cum = torch.cat([torch.ones(batch_size, 1, device=predicted_relevance.device), prod_cum[:, :-1]], dim=1)
    
    # Compute ERR: Sum over (1 / i) * r_i * prod_cum[:, i]
    weights = 1 / torch.arange(1, top_m + 1, device=predicted_relevance.device, dtype=torch.float32)
    err_pred = torch.sum(weights * predicted_relevance_partial * prod_cum, dim=1).cpu()
    
    # rr true
    rr_true = dict_embed['rr_true'].cpu()

    ''' Compute DCG '''
    discounts = torch.log2(torch.arange(2, top_k + 2).float().to(predicted_relevance.device))
    gains = predicted_relevance
    dcg_pred = (gains / discounts).sum(dim=-1).cpu()

    # Compute ideal DCG (IDCG) by sorting relevance ideally (highest scores first)
    ideal_relevance, _ = torch.sort(predicted_relevance, dim=-1, descending=True)
    ideal_gains = ideal_relevance
    idcg_pred = (ideal_gains / discounts).sum(dim=-1).cpu()

    # Compute ndcg
    ndcg_pred = (dcg_pred / idcg_pred).cpu()
    ndcg_pred[idcg_pred == 0] = 0

    # ndcg true
    ndcg_true = dict_embed['ndcg_true'].cpu()

    if corr_bool:

        # calculate corr
        if target_metric == 'ndcg@10':
            dcg_ndcg_pear_coef, _ = pearsonr(dcg_pred, ndcg_true)
            dcg_ndcg_kend_coef, _ = kendalltau(dcg_pred, ndcg_true)
            ndcg_ndcg_pear_coef, _ = pearsonr(ndcg_pred, ndcg_true)
            ndcg_ndcg_kend_coef, _ = kendalltau(ndcg_pred, ndcg_true)
        elif target_metric == 'mrr@10':
            dcg_ndcg_pear_coef, _ = pearsonr(err_pred, rr_true)
            dcg_ndcg_kend_coef, _ = kendalltau(err_pred, rr_true)
            ndcg_ndcg_pear_coef, _ = pearsonr(err_pred, rr_true)
            ndcg_ndcg_kend_coef, _ = kendalltau(err_pred, rr_true)
        else:
            raise ValueError("target_metric은 'ndcg@10' 또는 'mrr@10'이어야 합니다.")

        return dcg_ndcg_pear_coef, dcg_ndcg_kend_coef, ndcg_ndcg_pear_coef, ndcg_ndcg_kend_coef
    else:
        return err_pred, rr_true, dcg_pred, ndcg_pred, ndcg_true


def infer_threshold_result(args, base_model, method, save_df=False):
    # base_model = 'ance'
    # base_model = 'bm25'
    # method = 'optimal'
    # method = 'train'
    # method = 'switch'
    # method = 'correct'

    dataset_list = ['msmarcodev', 'DL2019', 'DL2020', 'DLHard']
    thres_type_list = ['no_thres', 'bayes', 'youden', 'f1', 'mean', 'median', 'quantile']
    # thres_type = 'no_thres'
    # thres_type = 'bayes'
    # thres_type = 'youden'
    # thres_type = 'median'
    # thres_type = 'f1'
    # dataset_name = 'msmarcotrain'
    # dataset_name = 'msmarcodev'
    # dataset_name = 'DL2019'
    # dataset_name = 'DL2020'
    coef_dcg_list = []
    coef_ndcg_list = []
    for thres_type in thres_type_list:

        coef_dcg_list_ = []
        coef_ndcg_list_ = []
        for dataset_name in dataset_list:
            
            dict_embed, args_infer = load_QPP_output(args, dataset_name=dataset_name, base_model=base_model, split_id=1, load_dict=True)
            
            if thres_type == 'no_thres':
                # no_threshold
                args_infer.err = True
            
            # elif thres_type == 'bayes':
                
            #     args_infer = load_QPP_output(args, dataset_name=dataset_name, base_model=base_model, split_id=1, load_dict=False)
                
            #     thres_dir = f"{args_infer.checkpoint_path}thres_{args_infer.setup}/"
            #     args_infer.threshold = torch.load(os.path.abspath(f"{thres_dir}{thres_type}_thres.pkl"))
            #     args_infer.threshold = args_infer.threshold.cuda()
            #     args_infer.err = False
                
            else:
                
                # set threshold
                if method == 'optimal':
                    args_infer = load_QPP_output(args, dataset_name=dataset_name, base_model=base_model, split_id=1, load_dict=False)

                elif method in ['train', 'correct']:
                    args_infer = load_QPP_output(args, dataset_name='msmarcotrain', base_model=base_model, split_id=1, load_dict=False)

                elif method == 'switch':
                    if dataset_name == 'DL2019':
                        args_infer = load_QPP_output(args, dataset_name='DL2020', base_model=base_model, split_id=1, load_dict=False)
                    elif dataset_name == 'DL2020':
                        args_infer = load_QPP_output(args, dataset_name='DL2019', base_model=base_model, split_id=1, load_dict=False)
                    elif dataset_name == 'DLHard':
                        args_infer = load_QPP_output(args, dataset_name='DLHard_valid', base_model=base_model, split_id=1, load_dict=False)
                    elif dataset_name == 'msmarcodev':
                        args_infer = load_QPP_output(args, dataset_name='msmarcotrain', base_model=base_model, split_id=1, load_dict=False)

                thres_dir = f"{args_infer.checkpoint_path}thres_{args_infer.setup}/"
                args_infer.threshold = torch.load(os.path.abspath(f"{thres_dir}{thres_type}_thres.pkl"))

                if method == 'correct':
                    if dataset_name in ['DL2019', 'DL2020', 'DLHard', 'DLHard_valid']:
                        ar = (dict_embed['actual_relevance'] >= 2).int().cpu()
                    else:
                        ar = dict_embed['actual_relevance'].int().cpu()
                    
                    p1 = ar.float().mean(dim=0)
                    # p1 = dict_embed['predicted_relevance'].mean(dim=0).cpu()

                    if base_model == 'bm25':
                        w1 = np.array([0.095, 0.061, 0.051, 0.036, 0.029, 0.027, 0.023, 0.020, 0.018, 0.015])
                    elif base_model == 'ance':
                        w1 = np.array([0.195, 0.110, 0.072, 0.058, 0.042, 0.035, 0.028, 0.025, 0.021, 0.020])
                    
                    alpha_correct_term = np.log(p1 * (1-w1) / ((1-p1) * (w1)))
                    args_infer.threshold = 1 / (1 +np.exp(-(np.log(args_infer.threshold / (1 - args_infer.threshold)) - alpha_correct_term)))
                    
                args_infer.threshold = args_infer.threshold.cuda()
                args_infer.err = False
            
            
            if dataset_name in ['DL2019', 'DL2020', 'DLHard', 'DLHard_valid']:
                args_infer.target_metric = 'ndcg@10'
            else:
                args_infer.target_metric = 'mrr@10'

            # dcg_ndcg_pear_coef, dcg_ndcg_kend_coef, _, _ = calculate_corr(dict_embed, args_infer.threshold, top_k=args_infer.top_k, err_bool=args_infer.err)
            dcg_ndcg_pear_coef, dcg_ndcg_kend_coef, ndcg_ndcg_pear_coef, ndcg_ndcg_kend_coef = calculate_corr(dict_embed, args_infer.threshold, top_k=args_infer.top_k, target_metric=args_infer.target_metric, err_bool=args_infer.err)
            
            coef_dcg_list_.extend([dcg_ndcg_pear_coef, dcg_ndcg_kend_coef])
            coef_ndcg_list_.extend([ndcg_ndcg_pear_coef, ndcg_ndcg_kend_coef])
        
        coef_dcg_list.append(coef_dcg_list_)
        coef_ndcg_list.append(coef_ndcg_list_)
    
    df_coef_dcg = pd.DataFrame(coef_dcg_list)
    df_coef_ndcg = pd.DataFrame(coef_ndcg_list)

    columns = [f"{dataset}_{suffix}" for dataset in dataset_list for suffix in ['pear', 'kend']]
    index = [f"{thres_type}" for thres_type in thres_type_list]

    df_coef_dcg.columns = columns
    df_coef_dcg.index = index
    df_coef_ndcg.columns = columns
    df_coef_ndcg.index = index

    if save_df:
        df_coef_dcg.to_csv(os.path.abspath(f'/root/default/Neural-IR/output/df_result_dcg_{base_model}_{method}.csv'))
        df_coef_ndcg.to_csv(os.path.abspath(f'/root/default/Neural-IR/output/df_result_ndcg_{base_model}_{method}.csv'))


    return df_coef_dcg, df_coef_ndcg

# base_model = 'bm25'
# base_model = 'ance'
# method = 'optimal'
# method = 'train'
# method = 'switch'
# for method in ['train', 'optimal', 'correct', 'switch']:
#     for base_model in ['bm25', 'ance']:
#         df_coef_dcg, df_coef_ndcg = infer_threshold_result(args, base_model, method, save_df=True)

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default='QPP_MLC')
    parser.add_argument("--mode", type=str, default='pilot')
    parser.add_argument("--cross_valid", type=str2bool, default=False)
    parser.add_argument("--fold_num", type=int, default=5)
    parser.add_argument("--test_mode", type=str2bool, default=False)
    parser.add_argument("--target_metric", type=str, default='mrr@10')
    parser.add_argument("--base_model", type=str, default='')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--dataset_list", nargs='+', type=str, default='')

    parser.add_argument("--query_path", type=str, default='')
    parser.add_argument("--qrels_path", type=str, default='')
    parser.add_argument("--run_path", type=str, default='')
    parser.add_argument("--ap_path", type=str, default='')
    parser.add_argument("--index_path", type=str, default='')
    parser.add_argument("--checkpoint_path", type=str, default='')
    parser.add_argument("--setup", type=str, default='')

    parser.add_argument("--random_seed", type=int, default=11)
    parser.add_argument("--epoch_num", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default= 2e-5)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--interval_save", type=int, default=10)

    parser.add_argument("--data_chunk_size", type=int, default=10000)
    parser.add_argument("--num_split", type=int, default=50)

    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_m", type=int, default=10)
    parser.add_argument("--embed_model", type=str, default='bert_cross')

    parser.add_argument("--transformer", type=str2bool, default=True)
    parser.add_argument("--position_encode", type=str2bool, default=True)
    parser.add_argument("--trans_nhead", type=int, default=8)
    parser.add_argument("--trans_num_layers", type=int, default=1)

    parser.add_argument("--posi_weight", type=float, default=1.0)
    parser.add_argument("--err", type=str2bool, default=True)
    parser.add_argument("--bert_epoch", type=str2bool, default=False)

    try:
        args = parser.parse_args()
        print("terminal")
    except:
        args = parser.parse_args([])
        print("interactive")

        args.name = 'QPP_MLC'
        args.mode = 'normal'    # ['normal', 'pilot']
        args.cross_valid = False
        args.fold_num = 1
        ###
        ###
        args.test_mode = True
        ###
        ###
        args.target_metric = 'mrr@10'
        args.base_model = 'bm25'    # ['bm25', 'ance']
        args.dataset = 'msmarcotrain'
        args.dataset_list = ['msmarcodev', 'DL2019', 'DL2020', 'DLHard']
        
        args.random_seed = 11
        args.epoch_num = 1
        args.batch_size = 16
        args.lr = 2e-5
        args.interval = 10
        args.interval_save = 10

        args.top_k = 10 
        args.top_m = 10
        args.embed_model = 'bert_cross'

        args.transformer = True
        args.position_encode = True
        args.trans_nhead = 8
        args.trans_num_layers = 1
        
        args.posi_weight = 1.0
        args.err = True
        args.bert_epoch = False
        

    args.query_path = f'./datasets/TREC/{args.dataset}/queries_{args.dataset}.jsonl'
    args.qrels_path = f'./datasets/TREC/{args.dataset}/qrels_{args.dataset}.jsonl'
    args.run_path = f'./retrieval_results/{args.base_model}_{args.dataset}_result'
    args.ap_path = f'./output/actual_performance/{args.base_model}_{args.dataset}_actual_performance.json'
    args.index_path = './datasets/collections/lucene-index-msmarco-passage'
    args.checkpoint_path = f'./supervisedQPP/QPP_MLC/checkpoint/'
    args.setup = f"{args.base_model}_{args.dataset}_{args.name}_{args.target_metric}"
    args.setup_dataset = f"{args.base_model}_{args.dataset}_{args.target_metric}"

    set_random_seed(seed=args.random_seed)
    print(args)
    
    ''' training '''
    # training_QPP_MLC(args)
