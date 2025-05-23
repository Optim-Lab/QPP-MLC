# Query Performance Prediction for Dense Retriever

Contents
- [Environments Settings](#Environments-Settings)
- [Data Preparation](#Data-Preparation)
  - [Raw File Download](#Raw-File-Download) 
  - [Preprocessing](#Preprocessing)
  - [Indexing](#Indexing)
  - [Retrieval](#Retrieval)
- [QPP](#QPP)
  - [Predicted Performance](#Predicted-Performance)
  - [Evaluation QPP](#Evaluation-QPP)

Note that for ease of use, we already uploaded the predicted performance files for all QPP methods reported in our paper.

## Environments Settings
We recommend running all the things in a Linux environment. 
To set up the environment on a new server or machine, simply run:
```bash
bash setup.sh
```

## Data Preparation
Query performance prediction for dense retriever needs **query**, **corpus**, **BM25_index**, **retrieval results files**, and **actual performance files**.

### Raw File Download

#### TREC 데이터 다운받기
아래 커맨드를 이용하여 MSMARCO passage corpus를 다운로드 받을 수 있습니다. 8.8M 개의 passage이 존재하고 압축파일의 용량은 1.0GB, 풀어진 데이터는 2.9GB 입니다.
```bash
mkdir -p datasets/collections/msmarco-passage

wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz -P datasets/collections/msmarco-passage

tar xvfz datasets/collections/msmarco-passage/collection.tar.gz -C datasets/collections/msmarco-passage
```

### Preprocessing
corpus의 형식을 tsv에서 jsonl으로 바꾸고 query, qrels의 raw data 또한 jsonl 형식으로 바꿉ㄴ디ㅏ.
#### tsv 파일 jsonl로 변형
tsv 형식의 corpus 파일을 9개의 jsonl 형식의 파일로 분할하여 저장합니다. 3.2GB의 용량이 필요합니다.
```bash
python convert_collection_to_jsonl.py \
  --collection-path datasets/collections/msmarco-passage/collection.tsv \
  --output-folder datasets/collections/msmarco-passage/collection_jsonl
```

#### query preprocessing
4개의 TREC 데이터셋 **msmarcodev**, **DL2019**, **DL2020**, **DLHard**의 형식을 모두 jsonl으로 바꿉니다.
```bash
python data_load.py --path_raw ./datasets/TREC --dataset_list msmarcotrain msmarcodev DL2019 DL2020 DLHard
```

### Indexing
사전에 corpus에 대한 두가지 인덱싱 과정이 필요하다. 먼저 언어 모형 기반 QPP 모델을 위해 lucene를 이용한 인덱스가 있다. 이어서 dense retrieval을 위한 2개 base_model(DPR, ANCE)을 이용하여 corpus의 embedding을 faiss_index로 저장한다.
#### Indexing corpus by pyserini.index.lucene
사용자의 환경에 따라 threads 조정 가능. index 결과는 4.2GB 요구됨.
```bash
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input datasets/collections/msmarco-passage/collection_jsonl \
  --index datasets/collections/lucene-index-msmarco-passage \
  --generator DefaultLuceneDocumentGenerator \
  --threads 9 \
  --storePositions --storeDocvectors --storeRaw
```

#### Corpus index
DPR, ANCE를 이용한 corpus embedding 후 faiss index 저장. 각 base_model의 index는 26GB, 즉 52GB 요구됨.
```bash
python corpus_index.py --base_model ance
python corpus_index.py --base_model dpr
```

### Retrieval
faiss idnex를 이용하여 base_model 2개와 datasets 4개의 조합으로 총 8개의 retrieval_results를 생성.
#### Get retrieval results from dense retriever (DPR, ANCE)
```bash
python retrieval.py \
  --base_model_list dpr ance \
  --dataset_list msmarcotrain msmarcodev DL2019 DL2020 DLHard
```

BM25 index를 이용하여 BM25 base_model과 datasets 4개의 조합으로 총 4개의 retrieval_results를 생성.
#### Get retrieval results from sparse lexical retriever (BM25)
```bash
python bm25.py --base_model_list bm25 --dataset_list msmarcotrain msmarcodev DL2019 DL2020 DLHard
```

#### Evaluation retrieval
retrieval_results외 qrels을 이용하여 NDCG, MRR 등 target_metric 계산. 추후 QPP의 ground_truth label로 사용된다.
```bash
python evaluation_retrieval.py \
  --base_model_list bm25 dpr ance \
  --dataset_list msmarcotrain msmarcodev DL2019 DL2020 DLHard
```

## QPP

### Predicted Performance

#### Query performance prediction (using post-retrieval method)
```bash
python QPP_method/post_retrieval.py \
  --query_path ./datasets/TREC/DL2019/queries_DL2019.jsonl \
  --qrels_path ./datasets/TREC/DL2019/qrels_DL2019.jsonl \
  --run_path ./retrieval_results/dpr_DL2019_result \
  --index_path ./datasets/collections/lucene-index-msmarco-passage \
  --output_path ./output/predicted_performance

#   --query_path ./datasets/TREC/{dataset}/queries_{dataset}.jsonl \
#   --qrels_path ./datasets/TREC/{dataset}/qrels_{dataset}.jsonl \
#   --run_path ./retrieval_results/{base_model}_{dataset}_result \
#   --index_path ./datasets/collections/lucene-index-msmarco-passage \
#   --output_path ./output/predicted_performance
```

### Evaluation QPP

#### Evaluation QPP clarity
```bash
python evaluation_QPP.py \
  --ap_path ./output/actual_performance/dpr_DL2019_actual_performance.json \
  --pp_path ./output/predicted_performance/dpr_DL2019_clarity-score-k100 \
  --target_metric mrr@10

# python evaluation_QPP.py \
#   --ap_path ./output/actual_performance/{base_model}_{dataset}_actual_performance.json \
#   --pp_path ./output/predicted_performance/{base_model}_{dataset}_{name} \
#   --target_metric mrr@10

```