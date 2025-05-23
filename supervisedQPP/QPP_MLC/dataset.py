#%%
import os
import json
import torch
import glob
import pickle
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import logging
logging.disable(logging.WARNING)

#%%
class DatasetQPPCross(Dataset):
    def __init__(self, args, split_id):

        self.args = args
        self.input = []
        self.searcher = None
        self.tokenizer = None

        if self.args.test_mode:
            self.data_dir = os.path.join(os.getcwd(), f"datasets/dataset_bert/{self.args.embed_model}_top_10_ex/data_{self.args.setup_dataset}/")
        else:
            self.data_dir = os.path.join(os.getcwd(), f"datasets/dataset_bert/{self.args.embed_model}_top_10/data_{self.args.setup_dataset}/")
        
        data_path_list = sorted(glob.glob(f"{self.data_dir}data_*{self.args.setup_dataset}*"))

        if len(data_path_list) > 0:
            input_ = torch.load(data_path_list[split_id - 1])
            self.input.extend(input_)
        else:
            self._initialize_searcher()
            self._initialize_tokenizer()
            self.load()
            data_path_list = sorted(glob.glob(f"{self.data_dir}data_*{self.args.setup_dataset}*"))
            input_ = torch.load(data_path_list[split_id - 1])
            self.input.extend(input_)

    def _initialize_searcher(self):
        self.searcher = LuceneSearcher(os.path.abspath(self.args.index_path))
    
    def _initialize_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def load(self):
        
        ''' raw data load '''
        # load query (train)
        with open(self.args.query_path,'r') as f:
            query = json.load(f)
        
        # load qrels (train)
        with open(self.args.qrels_path,'r') as f:
            qrels = json.load(f)
    
        # load actual performance (train)
        with open(self.args.ap_path, 'r') as r:
            ap_bank = json.loads(r.read())
        
        # load run (result list), 2 min
        with open(self.args.run_path, 'rb') as f:
            run = pickle.load(f)

        if self.args.test_mode:
            run = dict(list(run.items())[:513])
            
        ''' make data directory '''
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        ''' generate data '''
        # qid = list(run.keys())[3]
        num_iter = 0
        for qid in tqdm(run.keys()):
            
            num_iter += 1
            
            if qid not in qrels:
                print(f"skip {qid}")
                continue

            if len(run[qid]) < self.args.top_k:
                print(f"skip {qid} : result list less then top_k")
                continue

            # qid에 대한 top_k개 pid와 score 생성
            pid_k, score = zip(*[(pid, score) for pid, score in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)[:self.args.top_k]])
            score = torch.tensor(score, dtype=float)

            # qid에 대한 relevance doc과 top_k doc의 관련성 생성
            qrels_posi = {k: v for k, v in qrels[qid].items() if v > 0}
            relevance_grades = torch.tensor([qrels_posi.get(key, 0) for key in pid_k])
            
            # searcher에서 doc의 text를 가져와 토크나이징
            query_pad = [query[qid] for _ in pid_k]
            doc_top_k = [json.loads(self.searcher.doc(str(pid)).raw())['contents'] for pid in pid_k]
            query_doc_top_k_pair = list(zip(query_pad, doc_top_k))

            token = self.tokenizer.batch_encode_plus(
                query_doc_top_k_pair,
                add_special_tokens=True,
                max_length=256,
                padding='max_length',
                return_tensors="pt",
                return_token_type_ids=True,
                return_attention_mask=True,
                truncation=True,
                return_special_tokens_mask=False,
            )
            
            # check encoded token by decode
            # decoded = tokenizer.decode(token["input_ids"][0], skip_special_tokens=False)
            
            ''' generate target metric (rr, ndcg) '''
            # Compute relevance array
            run_pids = list(run[qid].keys())[:self.args.top_m]
            rel_array = np.zeros(self.args.top_m)  # Initialize with zeros
            rel_array[:len(run_pids)] = [qrels_posi.get(pid, 0) for pid in run_pids]

            # Compute ideal relevance array
            sorted_items = sorted(qrels_posi.values(), reverse=True)
            rel_array_ideal = np.zeros(self.args.top_m)
            rel_array_ideal[:len(sorted_items[:self.args.top_m])] = sorted_items[:self.args.top_m]

            # Compute DCG and IDCG
            discounts = np.log2(np.arange(2, self.args.top_m + 2))
            dcg = (rel_array / discounts).sum()
            idcg = (rel_array_ideal / discounts).sum()

            # Compute nDCG@m
            ndcg = dcg / idcg if idcg > 0 else 0
            
            # Compute rr@m
            positions = torch.arange(1, self.args.top_m + 1).float()
            if self.args.dataset in ['msmarcotrain', 'msmarcodev']:
                relevance_binary = relevance_grades[:self.args.top_m] >= 1
            else:
                relevance_binary = relevance_grades[:self.args.top_m] >= 2

            rr = (relevance_binary / positions).max()

            # Compute actual performance
            ap = torch.tensor(float(ap_bank[qid][self.args.target_metric]))
            
            # input 
            self.input.append([
                qid,
                token["input_ids"],
                token["attention_mask"],
                token["token_type_ids"],
                relevance_grades.unsqueeze(0),
                score.unsqueeze(0),
                ap,
                rr,
                torch.tensor(dcg),
                torch.tensor(idcg),
                torch.tensor(ndcg)
                ])

            if num_iter % self.args.data_chunk_size == 0 and num_iter < self.args.num_split * self.args.data_chunk_size:
                split_id = num_iter // self.args.data_chunk_size
                
                print(split_id)

                data_path = f"{self.data_dir}data_{self.args.setup_dataset}_{split_id:02}.pkl"
                torch.save(self.input, data_path)
                self.input = []

            if num_iter == len(run):
                split_id = num_iter // self.args.data_chunk_size
                
                print(split_id)

                data_path = f"{self.data_dir}data_{self.args.setup_dataset}_{split_id:02}.pkl"
                torch.save(self.input, data_path)
                self.input = []
            
        run.clear()

    def __getitem__(self, index):
        qid, input_ids, attention_mask, token_type_ids, rg, score, ap, rr, dcg, idcg, ndcg = self.input[index]
        return [qid, input_ids, attention_mask, token_type_ids, rg, score, ap, rr, dcg, idcg, ndcg]
        
    def __len__(self):
        return len(self.input)



#%%
def collate_fn(data):
    qid, q_input_ids, q_attention_mask, q_token_type_ids, d_input_ids, d_attention_mask, d_token_type_ids, ap, rg, idcg = zip(*data)

    return {'qid': qid, #  tuple(batch)
            'q_input_ids': torch.cat(q_input_ids,0), # [batch, 512]
            'q_attention_mask':torch.cat(q_attention_mask,0),  # [batch, 512]
            'q_token_type_ids':torch.cat(q_token_type_ids,0),  # [batch, 512]
            'd_input_ids': torch.stack(d_input_ids),  # [batch, k, 512]
            'd_attention_mask': torch.stack(d_attention_mask),  # [batch, k, 512]
            'd_token_type_ids': torch.stack(d_token_type_ids),  # [batch, k, 512]
            'ap': torch.tensor(ap), # [batch]
            'rg': torch.cat(rg,0), # [batch, k]
            'idcg': torch.tensor(idcg), # [batch]
            }


def collate_fn_cross(data):
    qid, input_ids, attention_mask, token_type_ids, rg, score, ap, rr, dcg, idcg, ndcg = zip(*data)

    return {'qid': qid, #  tuple(batch)
            'input_ids': torch.stack(input_ids, 0), # [batch, k, len]
            'attention_mask':torch.stack(attention_mask, 0),  # [batch, k, len]
            'token_type_ids':torch.stack(token_type_ids, 0),  # [batch, k, len]
            'rg': torch.cat(rg, 0), # [batch, k]
            'score': torch.cat(score, 0), # [batch, k]
            'ap': torch.tensor(ap), # [batch]
            'rr': torch.tensor(rr), # [batch]
            'dcg': torch.tensor(dcg), # [batch]
            'idcg': torch.tensor(idcg), # [batch]
            'ndcg': torch.tensor(ndcg), # [batch]
            }
