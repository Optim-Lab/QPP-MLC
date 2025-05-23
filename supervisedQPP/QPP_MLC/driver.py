#%%
import os
import copy
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr, kendalltau
from dataset import DatasetQPPCross, collate_fn_cross

#%%
class Driver(object):
    def __init__(self, args, model):
        super(Driver, self).__init__()
        self.args = args

        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model

    def training(self, dataset, collate_fn_cross, epoch_id, split_id, fold_id, optimizer):
        
        ''' load train data '''
        self.model.train()

        dataloader = DataLoader(dataset, collate_fn=collate_fn_cross, batch_size=self.args.batch_size, shuffle=True)
        # data = next(iter(dataloader))
        
        ''' training '''
        step = 0
        total_num_q = 0
        loss_display = 0
        loss_total = 0
        start_time = time.time()

        for j, data in enumerate(dataloader, 0):
            if torch.cuda.is_available():
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_cuda[key] = value.cuda()
                    else:
                        data_cuda[key] = value
                data = data_cuda

            loss, _, _ = self.model(data, target_metric='mrr@10')
            loss_display += loss.item()
            loss_total += loss.item()
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            total_num_q += len(data['qid'])
            step += 1

            if step % self.args.interval == 0:
                elapsed_time = time.time() - start_time
                ms_per_query = elapsed_time / total_num_q * 1000

                print(f'Training: {self.args.setup}')
                print(f'Epoch_id:{epoch_id}, Split:{split_id}, Step:{step}, Loss:{loss_display/self.args.interval}, LR:{self.args.lr}, sec/q_train:{ms_per_query:.4f}')

                loss_display = 0
        
        loss_total = loss_total / step

        ''' checkpoint save '''
        weight_dir = f"{self.args.checkpoint_path}weight_{self.args.setup}/"
        
        if self.args.cross_valid:
            weight_path = f"{weight_dir}weight_{self.args.setup}_{epoch_id:02}_{split_id:02}_{fold_id:02}.pkl"
        else:
            weight_path = f"{weight_dir}weight_{self.args.setup}_{epoch_id:02}_{split_id:02}.pkl"
        
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        if split_id % self.args.interval_save == 0 or split_id == self.args.num_split:
            torch.save(self.model.state_dict(), os.path.abspath(weight_path))
            print("Saved the model trained on split {} ".format(split_id))

        ''' inference '''
        eval_metric_dict = {}
        eval_metric_dict_tilde = {}
        for dataset_name in self.args.dataset_list:

            eval_metric_dict_, eval_metric_dict_tilde_ = self.inference(dataset_name=dataset_name)
            eval_metric_dict.update(eval_metric_dict_)
            eval_metric_dict_tilde.update(eval_metric_dict_tilde_)

            print(dataset_name)
            print(eval_metric_dict_)
        
        result_dict = eval_metric_dict | {'loss_train' : loss_total}
        result_dict_tilde = eval_metric_dict_tilde | {'loss_train' : loss_total}

        return result_dict, result_dict_tilde
    

    def inference(self, dataset_name):

        ''' evaluation dataset load '''
        args_infer = copy.deepcopy(self.args)
        args_infer.dataset = dataset_name
        if dataset_name in ['DL2019', 'DL2020', 'DLHard']:
            args_infer.target_metric = 'ndcg@10'

        args_infer.run_path = f'./retrieval_results/{args_infer.base_model}_{args_infer.dataset}_result'
        args_infer.qrels_path = f'./datasets/TREC/{args_infer.dataset}/qrels_{args_infer.dataset}.jsonl'
        args_infer.query_path = f'./datasets/TREC/{args_infer.dataset}/queries_{args_infer.dataset}.jsonl'
        args_infer.ap_path = f'./output/actual_performance/{args_infer.base_model}_{args_infer.dataset}_actual_performance.json'
        # args_infer.setup = f"{args_infer.base_model}_{args_infer.dataset}_{args_infer.name}_{args_infer.target_metric}"
        args_infer.setup_dataset = f"{args_infer.base_model}_{args_infer.dataset}_{args_infer.target_metric}"
        args_infer.batch_size = int(32)

        dataset_infer = DatasetQPPCross(args_infer, split_id=1)
        dataloader_infer = DataLoader(dataset_infer, collate_fn=collate_fn_cross, batch_size=args_infer.batch_size, shuffle=False)
        
        ''' inference '''
        start_time = time.time()

        # model = model.cuda()
        self.model.eval()
        
        total_samples = len(dataloader_infer.dataset)
        batch_size = dataloader_infer.batch_size

        with torch.no_grad():
            pp_list = []
            ap_list = []
            ap_ours_list = []
            ndcg_tilde_list = []

            pp_list = torch.empty(total_samples, device='cuda')
            ap_list = torch.empty(total_samples, device='cuda')
            ap_ours_list = torch.empty(total_samples, device='cuda')
            ndcg_tilde_list = torch.empty(total_samples, device='cuda')

            step = 0
            loss_infer_total = 0
            for j, data in tqdm(enumerate(dataloader_infer, 0)):
                if torch.cuda.is_available():
                    data_cuda = dict()
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data_cuda[key] = value.cuda()
                        else:
                            data_cuda[key] = value
                    data = data_cuda
                
                ap_list[batch_size*j : batch_size*(j+1)] = data['ap']

                if args_infer.target_metric == 'mrr@10':
                    ap_ours_list[batch_size*j : batch_size*(j+1)] = data['rr']
                elif args_infer.target_metric == 'ndcg@10':
                    ap_ours_list[batch_size*j : batch_size*(j+1)] = data['ndcg']

                loss_infer, pp, ndcg_tilde = self.model(data, target_metric=args_infer.target_metric)

                pp_list[batch_size*j : batch_size*(j+1)] = pp

                ndcg_tilde_list[batch_size*j : batch_size*(j+1)] = ndcg_tilde

                step += 1
                loss_infer_total += loss_infer.item()
            
            loss_infer_total = loss_infer_total / step

            pp_list = pp_list.tolist()
            ap_list = ap_list.tolist()
            ap_ours_list = ap_ours_list.tolist()
            ndcg_tilde_list = ndcg_tilde_list.tolist()

            # pp_list = torch.rand(16).cuda().tolist()
            pearson_coef, pearson_p = pearsonr(ap_ours_list, pp_list)
            kendall_coef, kendall_p = kendalltau(ap_ours_list, pp_list)
            spearman_coef, spearman_p = spearmanr(ap_ours_list, pp_list)
            sMARE, _ = sARE(ap_ours_list, pp_list, rankType='average')
            
            eval_metric_dict = {
                f'pearson_coef_{dataset_name}' : pearson_coef,
                f'kendall_coef_{dataset_name}' : kendall_coef,
                f'spearman_coef_{dataset_name}' : spearman_coef,
                f'sMARE_{dataset_name}' : sMARE,
                f'loss_infer_{dataset_name}' : loss_infer_total,
            }
            
            # 
            pearson_coef_tilde, pearson_p_tilde = pearsonr(ap_ours_list, ndcg_tilde_list)
            kendall_coef_tilde, kendall_p_tilde = kendalltau(ap_ours_list, ndcg_tilde_list)
            spearman_coef_tilde, spearman_p_tilde = spearmanr(ap_ours_list, ndcg_tilde_list)
            sMARE_tilde, _ = sARE(ap_ours_list, ndcg_tilde_list, rankType='average')
            
            eval_metric_dict_tilde = {
                f'pearson_coef_{dataset_name}' : pearson_coef_tilde,
                f'kendall_coef_{dataset_name}' : kendall_coef_tilde,
                f'spearman_coef_{dataset_name}' : spearman_coef_tilde,
                f'sMARE_{dataset_name}' : sMARE_tilde,
            }
        
        elapsed_time = time.time() - start_time
        ms_per_query = elapsed_time / len(dataset_infer) * 1000
        print(f'ms/q_infer:{ms_per_query:.4f}')

        return eval_metric_dict, eval_metric_dict_tilde
    

    def embedding_check(self, dataloader_infer):

        self.model.eval()
        total_samples = len(dataloader_infer.dataset)
        batch_size = dataloader_infer.batch_size

        with torch.no_grad():
            pp_list = torch.empty(total_samples, device='cuda')
            ap_list = torch.empty(total_samples, device='cuda')

            for j, data in enumerate(dataloader_infer, 0):
                if torch.cuda.is_available():
                    data_cuda = dict()
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data_cuda[key] = value.cuda()
                        else:
                            data_cuda[key] = value
                    data = data_cuda
                
                ap_list[batch_size*j : batch_size*(j+1)] = data['ap']

                loss, pp, dict_embed = self.model(data, debug=True, target_metric=self.args.target_metric)
                pp_list[batch_size*j : batch_size*(j+1)] = pp

                if j==0:
                    combined_dict = {key: [] for key in dict_embed.keys()}
        
                for key, tensor in dict_embed.items():
                    combined_dict[key].append(tensor)
            dict_embed_total = {key: torch.cat(tensors, dim=0) for key, tensors in combined_dict.items()}

            pp_list = pp_list.tolist()
            ap_list = ap_list.tolist()

        return loss, pp_list, ap_list, dict_embed_total


class DriverThres(object):
    def __init__(self, args, model_soft_thres):
        super(DriverThres, self).__init__()
        self.args = args

        if torch.cuda.is_available():
            self.model_soft_thres = model_soft_thres.cuda()
        else:
            self.model_soft_thres = model_soft_thres

    def training(self, dict_embed, epoch_id, split_id, optimizer):
        
        ''' load train data '''
        self.model_soft_thres.train()

        len_q = len(dict_embed['qid_list'])

        ''' training '''
        step = 0
        total_num_q = 0
        loss_display = 0
        loss_total = 0
        start_time = time.time()

        for i in range(0, len_q, self.args.batch_size):
            batch_dict_embed = {
                key: value[i:i+self.args.batch_size] for key, value in dict_embed.items()
            }
            
            predicted_relevance = batch_dict_embed['predicted_relevance']
            dcg_true = batch_dict_embed['dcg_true'].float()

            R_tilde, loss_dcg, dcg_pred = self.model_soft_thres(R_hat=predicted_relevance, dcg_true=dcg_true)
        
            loss_display += loss_dcg.item()
            loss_total += loss_dcg.item()
            loss_dcg.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            total_num_q += len(batch_dict_embed['qid_list'])
            step += 1

            if step % self.args.interval == 0:
                elapsed_time = time.time() - start_time
                ms_per_query = elapsed_time / total_num_q * 1000

                print(f'Training: {self.args.setup}')
                print(f'Epoch_id:{epoch_id}, Split:{split_id}, Step:{step}, Loss:{loss_display/self.args.interval}, LR:{self.args.lr}, sec/q_train:{ms_per_query:.4f}')
                # print(self.model_soft_thres._parameters)

                loss_display = 0
        
        loss_total = loss_total / step

        thres_dict = {
            **{f"tau_{i+1}": self.model_soft_thres._parameters['tau'][i].item() for i in range(10)},
            **{f"beta_{i+1}": self.model_soft_thres._parameters['beta'][i].item() for i in range(10)},
            "loss_train" : loss_total,
        }

        ''' checkpoint save '''
        thres_dir = f"{self.args.checkpoint_path}thres_{self.args.setup}/"
        thres_path = f"{thres_dir}thres_{self.args.setup}_{epoch_id:02}_{split_id:02}.pkl"
        
        if not os.path.exists(thres_dir):
            os.makedirs(thres_dir)

        if split_id % self.args.interval_save == 0 or split_id == self.args.num_split:
            torch.save(self.model_soft_thres.state_dict(), os.path.abspath(thres_path))
            print("Saved the model_soft_thres trained on split {} ".format(split_id))

        ''' inference '''
        eval_metric_dict = {}
        for dataset_name in self.args.dataset_list:

            eval_metric_dict_ = self.inference(dataset_name=dataset_name)
            eval_metric_dict.update(eval_metric_dict_)
        
        result_dict = eval_metric_dict | thres_dict | {'loss_train' : loss_total}
        print(result_dict)

        return result_dict
    
    
    def inference(self, dataset_name):

        ''' evaluation dataset load '''
        # dataset_name = 'DL2019'
        # dataset_name = 'DL2020'
        # dataset_name = 'DLHard'
        # dataset_name = 'msmarcodev'
        args_infer = copy.deepcopy(self.args)
        args_infer.dataset = dataset_name
        if dataset_name in ['DL2019', 'DL2020', 'DLHard']:
            args_infer.target_metric = 'ndcg@10'

        args_infer.run_path = f'./retrieval_results/{args_infer.base_model}_{args_infer.dataset}_result'
        args_infer.qrels_path = f'./datasets/TREC/{args_infer.dataset}/qrels_{args_infer.dataset}.jsonl'
        args_infer.query_path = f'./datasets/TREC/{args_infer.dataset}/queries_{args_infer.dataset}.jsonl'
        args_infer.ap_path = f'./output/actual_performance/{args_infer.base_model}_{args_infer.dataset}_actual_performance.json'
        args_infer.setup = f"{args_infer.base_model}_{args_infer.dataset}_{args_infer.name}_{args_infer.target_metric}"
        args_infer.setup_dataset = f"{args_infer.base_model}_{args_infer.dataset}_{args_infer.target_metric}"
        args_infer.batch_size = int(32)


        data_dir = f"{args_infer.checkpoint_path}data_{args_infer.setup}/"
        data_path = f"{data_dir}data_{args_infer.setup_dataset}_01.pkl"
        dict_embed = torch.load(data_path)
        
        predicted_relevance = dict_embed['predicted_relevance']
        dcg_true = dict_embed['dcg_true'].float()
        R_tilde, loss_dcg, dcg_pred = self.model_soft_thres(R_hat=predicted_relevance, dcg_true=dcg_true)

        pearson_coef, pearson_p = pearsonr(dcg_true.tolist(), dcg_pred.tolist())
        kendall_coef, kendall_p = kendalltau(dcg_true.tolist(), dcg_pred.tolist())
        spearman_coef, spearman_p = spearmanr(dcg_true.tolist(), dcg_pred.tolist())
        sMARE, _ = sARE(dcg_true.tolist(), dcg_pred.tolist(), rankType='average')
        
        eval_metric_dict = {
            f'pearson_coef_{dataset_name}' : pearson_coef,
            f'kendall_coef_{dataset_name}' : kendall_coef,
            f'spearman_coef_{dataset_name}' : spearman_coef,
            f'sMARE_{dataset_name}' : sMARE,
            f'loss_infer_{dataset_name}' : loss_dcg,
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
