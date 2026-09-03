# Generalizing Query Performance Prediction under Retriever and Concept Shifts via Data-driven Correction

![QPP-MLC Architecture](./fig_QPP_arch5.png)


Contents
- [Environment Settings](#Environments-Settings)
- [Data Preparation](#Data-Preparation)
  - [Raw File Download](#Raw-File-Download) 
  - [Preprocessing](#Preprocessing)
  - [Indexing](#Indexing)
  - [Retrieval](#Retrieval)
- [Unsupervised QPP](#Unsupervised-QPP)
  - [Predicted Performance](#Predicted-Performance)
- [QPP-MLC](#QPP-MLC)
  - [Training QPP-MLC](#Training-QPP-MLC)
  - [Inference with QPP-MLC](#Inference-with-QPP-MLC)
  - [QPP-MLC-b](#QPP-MLC-b)

## Environment Settings
We recommend running all components	in a Linux environment. 
To set up the environment on a new server or machine, simply run:
```bash
bash setup.sh
```

## Data Preparation
Query performance prediction needs **query**, **corpus**, **BM25_index**, **ANCE_faiss_index**, **retrieval results files**, and **actual performance files**.

### Raw File Download

#### Downloading TREC Data
You can download the MS MARCO passage corpus using the command below.
The dataset contains approximately 8.8M passages.
The compressed file is 1.0 GB, and the extracted size is 2.9 GB.
```bash
mkdir -p datasets/collections/msmarco-passage

wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz -P datasets/collections/msmarco-passage

tar xvfz datasets/collections/msmarco-passage/collection.tar.gz -C datasets/collections/msmarco-passage
```

### Preprocessing
We convert the original corpus from TSV format to JSONL format. The raw query and qrels files are also converted to JSONL.

#### Converting TSV corpus to JSONL
The corpus in TSV format is split into 9 JSONL files.  
This step requires approximately 3.2 GB of disk space.
```bash
python convert_collection_to_jsonl.py \
  --collection-path datasets/collections/msmarco-passage/collection.tsv \
  --output-folder datasets/collections/msmarco-passage/collection_jsonl
```

#### Query preprocessing
We convert the query files from four TREC datasets - **msmarcodev**, **DL2019**, **DL2020**, and **DLHard** - into JSONL format.
```bash
python data_load.py --path_raw ./datasets/TREC --dataset_list msmarcotrain msmarcodev DL2019 DL2020 DLHard
```

### Indexing
We perform two types of indexing on the corpus before running the QPP models.  
First, we build a **Lucene index** using Pyserini for language model-based QPP models.  
Next, for dense retrieval, we generate FAISS indexes of the corpus embeddings using **ANCE**.

#### Indexing the corpus with pyserini.index.lucene
You can adjust the number of threads based on your system environment.  
The resulting Lucene index requires approximately 4.2 GB of disk space.

```bash
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input datasets/collections/msmarco-passage/collection_jsonl \
  --index datasets/collections/lucene-index-msmarco-passage \
  --generator DefaultLuceneDocumentGenerator \
  --threads 9 \
  --storePositions --storeDocvectors --storeRaw
```

#### Corpus indexing
We embed the corpus using **ANCE** and store the result as a FAISS index.  
This process requires approximately **52 GB** of disk space.
```bash
python corpus_index.py --base_model ance
```

### Retrieval
We generate retrieval results for all combinations of retrievers (BM25, ANCE) and datasets (msmarcotrain, msmarcodev, DL2019, DL2020, DLHard).

#### Get retrieval results from dense retriever (ANCE)
```bash
python retrieval.py \
  --base_model_list ance \
  --dataset_list msmarcotrain msmarcodev DL2019 DL2020 DLHard
```

#### Get retrieval results from sparse lexical retriever (BM25)
```bash
python bm25.py --base_model_list bm25 --dataset_list msmarcotrain msmarcodev DL2019 DL2020 DLHard
```

#### Evaluate Retrieval Results
We compute target metrics such as nDCG and MRR using the retrieval results and the corresponding qrels.
These metric values will later serve as ground-truth labels for supervised QPP training.
```bash
python evaluation_retrieval.py \
  --base_model_list bm25 dpr ance \
  --dataset_list msmarcotrain msmarcodev DL2019 DL2020 DLHard
```

## Unsupervised QPP

### Predicting Query Performance (Unsupervised)

#### Query performance prediction (using unsupervised post-retrieval method)
Run post-retrieval QPP methods (e.g., Clarity, NQC) on the retrieval results:
```bash
python unsupervisedQPP/post_retrieval.py \
  --base_model_list bm25 ance \
  --dataset_list msmarcotrain msmarcodev DL2019 DL2020 DLHard
```

#### Evaluate Unsupervised QPP
Compute correlations (e.g., Pearson, Kendall) between predicted and actual performance:
```bash
python evaluation_QPP.py \
  --base_model_list bm25 ance \
  --dataset_list msmarcodev DL2019 DL2020 DLHard
```
## QPP-MLC (Supervised)

### Training QPP-MLC
Train the multi-label classification (MLC) model on the msmarcotrain dataset:
```bash
python supervisedQPP/QPP_MLC/main.py \
  --name QPP_MLC \
  --mode normal \
  --base_model bm25 \
  --dataset msmarcotrain \
  --dataset_list msmarcodev DL2019 DL2020 DLHard \
  --batch_size 16 \
  --lr 2e-5 \
  --top_k 10 \
  --top_m 10 \
  --embed_model bert_cross \
  --trans_nhead 8 \
  --trans_num_layers 1 \
  --class_weight one \
  --posi_weight 1.0 \
  --err True \
  --threshold 0.5 \
  --action training
```

### Inference with QPP-MLC
Predict query performance on new datasets using the trained MLC model:
```bash
python supervisedQPP/QPP_MLC/main.py \
  --name QPP_MLC \
  --mode normal \
  --base_model bm25 \
  --dataset msmarcotrain \
  --base_model_list bm25 ance \
  --dataset_list msmarcodev DL2019 DL2020 DLHard \
  --batch_size 16 \
  --lr 2e-5 \
  --top_k 10 \
  --top_m 10 \
  --embed_model bert_cross \
  --trans_nhead 8 \
  --trans_num_layers 1 \
  --class_weight one \
  --posi_weight 1.0 \
  --err True \
  --threshold 0.5 \
  --action inference
```

### QPP-MLC-b

#### Extract Embeddings and Metadata
Generates embedding and metadata files from the trained QPP-MLC model.  
The output will be stored in: ./supervisedQPP/QPP_MLC/checkpoint/data_{base_model}_{dataset}_QPP_MLC_{target_metric}
```bash
python supervisedQPP/QPP_MLC/thres.py --action embed
```

#### Generate Thresholds
Computes optimal thresholds for binarizing predicted relevance scores.
The output will be saved in: ./supervisedQPP/QPP_MLC/checkpoint/thres_{base_model}_{dataset}_QPP_MLC_{target_metric}
```bash
python supervisedQPP/QPP_MLC/thres.py --action gener_thres
```

#### Generate Prediction Results
Evaluates QPP-MLC-b predictions using computed thresholds and generates result tables.  
The output will be saved in: ./output/
```bash
python supervisedQPP/QPP_MLC/thres.py --action gener_result
```

## Acknowledgement
This repository was developed with support from the **서울시립대학교 데이터 사이언스 플러스 차세대 융합인재 양성사업단** - http://dsplus.uos.ac.kr/
