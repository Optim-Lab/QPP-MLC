#%%
import easydict
import argparse
import os
import json
import pathlib
import logging
import torch
import faiss
import faiss.contrib.torch_utils
import numpy as np
from tqdm import tqdm
from utils_bert import SentenceBERT

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#%%
def corpus_index(corpus, args):
    
    base_model_dict = {
        'ance' : "msmarco-roberta-base-ance-firstp",
        'dpr' : ("facebook-dpr-question_encoder-multiset-base",
                "facebook-dpr-ctx_encoder-multiset-base",
                " [SEP] "),
    }
    
    model_base = SentenceBERT(base_model_dict[args.base_model], batch_size=args.batch_size)
    
    corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
    corpus_ids_numeric = [int(id_str) for id_str in corpus_ids]
    corpus_list = [corpus[str(cid)] for cid in corpus_ids_numeric]

    del corpus, corpus_ids

    itr = range(0, len(corpus_list), args.corpus_chunk_size)
    
    # corpus_embeddings = torch.zeros([len(corpus_list), model_base.doc_model[1].word_embedding_dimension]) # d=768

    faiss_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "corpus_faiss/{}_{}_faiss/".format(args.base_model, args.corpus_name))

    if not os.path.exists(faiss_dir):
        os.makedirs(faiss_dir)

    index_flat = faiss.IndexFlatIP(model_base.doc_model[1].word_embedding_dimension)
    index = faiss.IndexIDMap2(index_flat)
    
    corpus_start_idx = 0
    corpus_end_idx = 1000

    for corpus_start_idx in itr:
        # logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
        corpus_end_idx = min(corpus_start_idx + args.corpus_chunk_size, len(corpus_list))
        
        # Encode chunk of corpus    
        sub_corpus_embeddings = model_base.encode_corpus(
            corpus_list[corpus_start_idx:corpus_end_idx],
            batch_size = args.batch_size,
            show_progress_bar = args.show_progress_bar,
            convert_to_tensor = args.convert_to_tensor
            ).to(device='cpu')
        
        index.add_with_ids(np.array(sub_corpus_embeddings), np.array(corpus_ids_numeric[corpus_start_idx:corpus_end_idx]))
        
    # torch.save(corpus_embeddings, os.path.join(faiss_dir, "{}_{}_embeddings.pt".format(args.base_model, args.corpus_name)))
    faiss.write_index(index, os.path.join(faiss_dir, "{}_{}_faiss".format(args.base_model, args.corpus_name)))

    return None


def query_index(args, dataset_name):
    
    base_model_dict = {
        'ance' : "msmarco-roberta-base-ance-firstp",
        'dpr' : ("facebook-dpr-question_encoder-multiset-base",
                "facebook-dpr-ctx_encoder-multiset-base",
                " [SEP] "),
    }
    
    model_base = SentenceBERT(base_model_dict[args.base_model], batch_size=args.batch_size)

    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets/TREC/")

    # dataset_name = 'DL2019'
    with open(f'{out_dir}{dataset_name}/queries_{dataset_name}.jsonl','r') as f:
        queries = json.load(f)
    

    queries_list = [queries[qid] for qid in queries]
    query_ids_numeric = [int(id_str) for id_str in list(queries.keys())]

    itr = range(0, len(queries_list), args.corpus_chunk_size)
    
    queries_embeddings = torch.zeros([len(queries_list), model_base.doc_model[1].word_embedding_dimension]) # d=768

    faiss_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), f"corpus_faiss/{args.base_model}_query_faiss/")

    if not os.path.exists(faiss_dir):
        os.makedirs(faiss_dir)

    index_flat = faiss.IndexFlatIP(model_base.doc_model[1].word_embedding_dimension)
    index = faiss.IndexIDMap2(index_flat)
    
    queries_start_idx = 0
    queries_end_idx = 1000

    for queries_start_idx in itr:
        # logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
        queries_end_idx = min(queries_start_idx + args.corpus_chunk_size, len(queries_list))
        
        # Encode chunk of quries
        sub_query_embeddings = model_base.encode_queries(
            queries_list[queries_start_idx:queries_end_idx],
            batch_size=args.batch_size,
            show_progress_bar=args.show_progress_bar,
            convert_to_tensor=args.convert_to_tensor
            ).to(device='cpu')
        
        index.add_with_ids(np.array(sub_query_embeddings), np.array(query_ids_numeric[queries_start_idx:queries_end_idx]))
        # index.add_with_ids(sub_corpus_embeddings.to(device='cpu'), corpus_ids[corpus_start_idx:corpus_end_idx])

        queries_embeddings[queries_start_idx:queries_end_idx] = sub_query_embeddings
        
    torch.save(queries_embeddings, os.path.join(faiss_dir, f"{args.base_model}_{dataset_name}_embeddings.pt"))
    faiss.write_index(index, os.path.join(faiss_dir, f"{args.base_model}_{dataset_name}_faiss"))

    return None

#%%
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_name", type=str, default='msmarco')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--corpus_chunk_size", type=int, default=50000)
    parser.add_argument("--show_progress_bar", type=bool, default=True)
    parser.add_argument("--convert_to_tensor", type=bool, default=True)
    parser.add_argument("--base_model", type=str, required=True)
    
    try:
        args = parser.parse_args()
    except:
        args = easydict.EasyDict({
            'corpus_name' : 'msmarco',
            'batch_size' : 256,
            'corpus_chunk_size' : 50000,
            'show_progress_bar' : True,
            'convert_to_tensor' : True,
            'base_model' : '',
        })
        args.base_model = 'ance'

    # load corpus
    corpus = {}
    path_collections = f'./datasets/collections/msmarco-passage/collection_jsonl'
    for i in range(9):
        file_path = os.path.join(path_collections, f'docs{i:02d}.json')
        
        if os.path.exists(file_path):
            with open(file_path, encoding='utf8') as fIn:
                for line in tqdm(fIn):
                    line = json.loads(line)
                    corpus[line.get("id")] = {"text": line.get("contents")}
    
    logger.info('Base model : ' + str(args.base_model))
    logger.info('Start indexing corpus with base model ' + str(args.base_model))
    corpus_index(corpus, args)


    # for dataset_name in ['msmarcotrain', 'msmarcodev', 'DL2019', 'DL2020', 'DLHard']:
    #     query_index(args, dataset_name=dataset_name)
    