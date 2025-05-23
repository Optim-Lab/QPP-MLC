#%%
import argparse
import easydict
import json
import math
import faiss
import logging
import pickle
import pathlib, os
from tqdm import tqdm
from utils_bert import *

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


#%%
class DenseRetrieval():
    
    def __init__(self, model_base, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):

        self.model_base = model_base
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = {}
    
    def search(self, faiss_index, query_ids, query_embeddings, top_k=1000):
        # logger.info("Search using faiss index!!")
        self.results = {qid: {} for qid in query_ids}
        sim, corpus_ids = faiss_index.search(np.array(query_embeddings), top_k)

        for i, qid in enumerate(query_ids):
            # score = sim[i].astype(float)
            score = sim[i].tolist()
            corpus_id = corpus_ids[i].astype(str)
            score_dict = dict(zip(corpus_id, score))
            self.results[qid] = score_dict

        return self.results
    
    def read_index_faiss(self, faiss_dir, base_model, corpus_name='msmarco', use_gpu=False):
        
        faiss_index = faiss.read_index(os.path.join(faiss_dir, "{}_{}_faiss".format(base_model, corpus_name)))
        logger.info("load faiss index(cpu)")
        
        if use_gpu:
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()  # use a single GPU
                faiss_index_gpu = faiss.index_cpu_to_gpu(res, 0, faiss_index)
                logger.info("move index from cpu to gpu")

                return faiss_index_gpu
            
            else:
                logger.info("cuda is not available")

        return faiss_index

    def get_queries_embeddings(self, queries):
        
        # logger.info("Encoding Queries...")

        queries_list = [queries[qid] for qid in queries]
        query_embeddings = self.model_base.encode_queries(
            queries_list,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=self.convert_to_tensor
            ).to(device='cpu')

        return query_embeddings
    

def dense_retriever(args, base_model, dataset, valid=False):
    
    base_model_dict = {
        'ance' : "msmarco-roberta-base-ance-firstp",
        'dpr' : ("facebook-dpr-question_encoder-multiset-base",
                "facebook-dpr-ctx_encoder-multiset-base",
                " [SEP] "),
    }
    
    model_base = SentenceBERT(base_model_dict[base_model], batch_size=args.batch_size)

    model = DenseRetrieval(
        model_base = model_base,
        batch_size = args.batch_size,
        corpus_chunk_size = args.corpus_chunk_size
        )

    faiss_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "corpus_faiss/{}_{}_faiss/".format(base_model, args.corpus_name))
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets/TREC/")
    
    if args.dataset == 'DLHard_valid':
        
        dataset_tmp = 'DL2019'
        with open('{}{}/queries_{}.jsonl'.format(out_dir, dataset_tmp, dataset_tmp),'r') as f:
            query_19 = json.load(f)
        
        dataset_tmp = 'DL2020'
        with open('{}{}/queries_{}.jsonl'.format(out_dir, dataset_tmp, dataset_tmp),'r') as f:
            query_20 = json.load(f)
        
        dataset_tmp = 'DLHard'
        with open('{}{}/queries_{}.jsonl'.format(out_dir, dataset_tmp, dataset_tmp),'r') as f:
            query_hard = json.load(f)

        q19 = list(query_19.keys())
        q20 = list(query_20.keys())
        qhard = list(query_hard.keys())

        qhard_diff = (set(q19) | set(q20)) - set(qhard)
        query_all = {**query_19, **query_20}
        queries = {qid: query_all[qid] for qid in qhard_diff if qid in query_all}

    else:
        with open('{}{}/queries_{}.jsonl'.format(out_dir, dataset, dataset),'r') as f:
            queries = json.load(f)
    
    if valid and args.dataset == 'msmarcotrain':
        queries = dict(list(queries.items())[:5000])
    
    query_ids = list(queries.keys())
    query_embeddings = model.get_queries_embeddings(queries)
    
    results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "retrieval_results")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    faiss_index = model.read_index_faiss(faiss_dir, base_model, args.corpus_name, use_gpu=True)
    itr = range(0, query_embeddings.shape[0], args.batch_size_faiss)
    results = {}
    
    for start_idx in tqdm(itr):
        
        end_idx = min(start_idx + args.batch_size_faiss, query_embeddings.shape[0])
        
        results_sub = model.search(
            faiss_index,
            query_ids[start_idx:end_idx],
            query_embeddings[start_idx:end_idx],
            top_k=args.top_k
            )
    
        results.update(results_sub)

        if dataset == 'msmarcotrain':
            if end_idx % args.query_chunk_size == 0 or end_idx == query_embeddings.shape[0]:
                
                num_split = int(end_idx / args.query_chunk_size)
                if end_idx == query_embeddings.shape[0]:
                    num_split = int(math.ceil(end_idx / args.query_chunk_size))
                
                num_split = str(num_split).zfill(2)
                
                if valid:
                    results_file = "{}/{}_{}_result_valid".format(results_dir, base_model, dataset)
                else:
                    results_file = "{}/{}_{}_result_{}".format(results_dir, base_model, dataset, num_split)
                
                with open(results_file, 'wb') as f:
                    pickle.dump(results, f)
                
                print('Save {} file'.format(results_file.split('/')[-1]))
                results.clear()
                # results = {}
    
    if dataset != 'msmarcotrain':

        if valid:
            results_file = "{}/{}_{}_result_valid".format(results_dir, base_model, dataset)
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)

        else:
            results_file = "{}/{}_{}_result".format(results_dir, base_model, dataset)
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)

    return results


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_name", type=str, default='msmarco')
    parser.add_argument("--score_function", type=str, default='dot')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--batch_size_faiss", type=int, default=10)
    parser.add_argument("--corpus_chunk_size", type=int, default=50000)
    parser.add_argument("--query_chunk_size", type=int, default=50000)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--base_model_list", nargs='+', type=str, required=True)
    parser.add_argument("--dataset_list", nargs='+', type=str, required=True)
    
    try:
        args = parser.parse_args()
    except:
        args = easydict.EasyDict({
            'corpus_name' : 'msmarco',
            'score_function' : 'dot',
            'batch_size' : 256,
            'batch_size_faiss' : 10,
            'corpus_chunk_size' : 50000,
            'query_chunk_size' : 50000,
            'top_k' : 1000,
            'base_model_list' : '',
            'dataset_list' : '',
        })
    
    for base_model in args.base_model_list:
        for dataset in args.dataset_list:
            args.dataset = dataset
            results = dense_retriever(args, base_model, dataset)
    