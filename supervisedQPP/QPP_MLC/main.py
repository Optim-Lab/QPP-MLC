#%%
import os
import sys
import glob
import torch
import argparse

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


def inference_QPP_MLC(driver, dataset_name):

    ''' inference '''
    # dataset_name = 'msmarcodev' # ['msmarcodev','DL2019','DL2020','DLHard']
    eval_metric_dict, _ = driver.inference(dataset_name=dataset_name)
    
    print(eval_metric_dict)

    return eval_metric_dict

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default='QPP_MLC')
    parser.add_argument("--mode", type=str, default='normal')
    parser.add_argument("--cross_valid", type=str2bool, default=False)
    parser.add_argument("--fold_num", type=int, default=1)
    parser.add_argument("--test_mode", type=str2bool, default=False)
    parser.add_argument("--target_metric", type=str, default='mrr@10')
    parser.add_argument("--base_model", type=str, default='')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--base_model_list", nargs='+', type=str, default=[])
    parser.add_argument("--dataset_list", nargs='+', type=str, default=[])

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

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_m", type=int, default=10)
    parser.add_argument("--embed_model", type=str, default='bert_cross')

    parser.add_argument("--transformer", type=str2bool, default=True)
    parser.add_argument("--position_encode", type=str2bool, default=True)
    parser.add_argument("--trans_nhead", type=int, default=8)
    parser.add_argument("--trans_num_layers", type=int, default=1)

    parser.add_argument("--posi_weight", type=float, default=1.0)
    parser.add_argument("--err", type=str2bool, default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bert_epoch", type=str2bool, default=False)

    parser.add_argument("--action", type=str, default='training')

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
        args.test_mode = False
        args.target_metric = 'mrr@10'
        args.base_model = 'bm25'    # ['bm25', 'ance']
        args.dataset = 'msmarcotrain'
        args.base_model_list = ['bm25', 'ance']
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
        args.threshold = 0.5
        args.bert_epoch = False

        args.action = 'inference'
        

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
    if args.action == 'training':
        result_dict = training_QPP_MLC(args)

    ''' inference '''
    if args.action == 'inference':
        for base_model in args.base_model_list:
            for dataset in args.dataset_list:
                
                args.base_model = base_model
                args.dataset = 'msmarcotrain'
                args.query_path = f'./datasets/TREC/{args.dataset}/queries_{args.dataset}.jsonl'
                args.qrels_path = f'./datasets/TREC/{args.dataset}/qrels_{args.dataset}.jsonl'
                args.run_path = f'./retrieval_results/{args.base_model}_{args.dataset}_result'
                args.ap_path = f'./output/actual_performance/{args.base_model}_{args.dataset}_actual_performance.json'
                args.index_path = './datasets/collections/lucene-index-msmarco-passage'
                args.checkpoint_path = f'./supervisedQPP/QPP_MLC/checkpoint/'
                args.setup = f"{args.base_model}_{args.dataset}_{args.name}_{args.target_metric}"
                args.setup_dataset = f"{args.base_model}_{args.dataset}_{args.target_metric}"

                ''' load fine-tuned model '''
                weight_dir = f"{args.checkpoint_path}weight_{args.setup}/"
                weight_path_list = sorted(glob.glob(f"{weight_dir}weight_*{args.setup}*"))
                split_id = len(weight_path_list)

                model = QPP_MLC(args)
                model.load_state_dict(torch.load(weight_path_list[split_id - 1]))
                driver = Driver(args, model)

                # args.base_model = 'bm25'
                # args.dataset = 'ance'
                args.base_model = base_model
                args.dataset = dataset
                args.setup = f"{args.base_model}_{args.dataset}_{args.name}_{args.target_metric}"
                args.setup_dataset = f"{args.base_model}_{args.dataset}_{args.target_metric}"
                eval_metric_dict = inference_QPP_MLC(driver, dataset_name=args.dataset)
