# KG-R3
Code for the CIKM'23 paper ["A Retrieve-and-Read Framework for Knowledge Graph Link Prediction"](https://arxiv.org/pdf/2212.09724.pdf) [ACM link](https://dl.acm.org/doi/10.1145/3583780.3614769)

<!-- Code will be released soon. -->

![A Retrieve-and-Read Framework for Knowledge Graph Link Prediction (KG-R3)](./assets/KG-R3.png)

## KG Link Prediction Results

|  Dataset  |   MRR  | HITS@1 | HITS@3 | HITS@10 |
|:---------:|:------:|:------:|:------:|:-------:|
| FB15K-237 | .390   | .315   | .413   | .539    |
| WN18RR    | .472   | .439   | .481   | .537    |


## Install dependencies
1. Create a new conda virtual env

2. Install horovod
```
HOROVOD_WITH_PYTORCH=1 --no-cache-dir --ignore-installed pip install horovod[pytorch] --extra-index-url https://download.pytorch.org/whl/cu113
```

3. Install other dependencies
```
pip install -r requirements.txt
```

## Download data

Download the preprocessed subgraphs and KG triples from [this link](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/pahuja_9_buckeyemail_osu_edu/ErHNYjTAzLZMgT7Mkgy1J_4BeoJMYTF4EQ2UxniOgPhCyA?e=85avhJ) from respective directories `FB15K-237` and `WN18RR` are place them in a `data/` directory.

## Dump retriever subgraphs (optional)

- Follow the steps in [dump_subgraphs.md](./dump_subgraphs.md) for details.

## Preprocess data

### pickle dataloader batches for faster training

#### FB15K-237, Minerva retriever
```
python -u dump_preproc_data.py --dataset-path data/FB15K-237/ \
--sampling-type minerva \
--batch-size 256 --out-dir data/FB15K-237/train_preproc/ \
--graph-connection type_1 --split train
```
- For WN18RR, set `--batch-size 512 --beam-size 40 --add-segment-embed --add-inverse-rels`

## Training

### FB15K-237

#### train, Minerva retriever
```
python -u main.py --dataset-path data/FB15K-237/ --cuda \
--save-dir ckpts/CKPT_DIR/ --sampling-type minerva \
--lr 1e-2 --warmup 0.1 --batch-size 512 \
--n-epochs 300 --patience 20 \
--seed 12548 > ckpts/CKPT_DIR/log.txt 2>&1
```
- For BFS retriever (FB15K-237 dataset), set `--sampling-type bfs --sample-size 100 --neigh-size 10`
- For one-hop neighborhood retriever (FB15K-237 dataset), set `--sampling-type onehop --sample-size 50`

### WN18RR

#### train, Minerva retriever
```
python -u main.py --dataset-path data/WN18RR/ --cuda \
--save-dir ckpts/CKPT_DIR/ --sampling-type minerva \
--lr 0.00175 --label-smoothing 0.1 --warmup 0.1 \
--batch-size 256 --n-epochs 500 \
--patience 100 --beam-size 40 --add-segment-embed --add-inverse-rels \
--seed 12548 > ckpts/CKPT_DIR/log.txt 2>&1
```
- For BFS retriever (WN18RR dataset), set `--sampling-type bfs --sample-size 30 --neigh-size 10 --lr 0.001`
- For one-hop neighborhood retriever (WN18RR dataset), set `--sampling-type onehop --sample-size 12 --lr 0.0004`

## Evaluation (specify split)
```
python eval.py --dataset-path <DATA_PATH> --cuda \
--ckpt-path ckpts/CKPT_DIR/model.pt \
--split <valid/test> --sampling-type minerva \
--graph-connection type_1  \
[--beam-size <>] [--add-segment-embed] [--add-inverse-rels]
```


## Citation
```
@inproceedings{10.1145/3583780.3614769,
author = {Pahuja, Vardaan and Wang, Boshi and Latapie, Hugo and Srinivasa, Jayanth and Su, Yu},
title = {A Retrieve-and-Read Framework for Knowledge Graph Link Prediction},
year = {2023},
isbn = {9798400701245},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3583780.3614769},
doi = {10.1145/3583780.3614769},
pages = {1992–2002},
numpages = {11},
location = {Birmingham, United Kingdom},
series = {CIKM '23}
}              
```
