# KG-R3
Code for the CIKM'23 paper ["A Retrieve-and-Read Framework for Knowledge Graph Link Prediction"](https://arxiv.org/pdf/2212.09724.pdf)

<!-- Code will be released soon. -->

![A Retrieve-and-Read Framework for Knowledge Graph Link Prediction (KG-R3)](./assets/KG-R3.png)

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

## Preprocess data

### pickle dataloader batches for faster training
```
python -u dump_preproc_data.py --dataset-path data/FB15K-237/ \
--sampling-type minerva \
--batch-size 512 --out-dir data/FB15K-237/train_preproc/ \
--graph-connection type_1 --split train --mode train
```

## Training

### FB15K-237

#### train, Minerva retriever
```
python -u main.py --dataset-path data/FB15K-237/ --cuda \
--save-dir ckpts/CKPT_DIR/ --sampling-type minerva \
--embed-dim 320 --n-attn-heads 8 --n-bert-layers 3 \
--lr 1e-2 --warmup 0.1 --batch-size 512 \
--n-epochs 300 --optimizer-type adamax --patience 20 \
--seed 12548 > ckpts/CKPT_DIR/log.txt 2>&1
```
- For BFS and one-hop neighborhood retrievers, change the `sampling-type` argument to `bfs` and `onehop` respectively.

### WN18RR

#### train, Minerva retriever
```
python -u main.py --dataset-path data/WN18RR/ --cuda \
--save-dir ckpts/CKPT_DIR/ --sampling-type minerva \
--embed-dim 320 --n-attn-heads 8 --n-bert-layers 3 \
--lr 0.00175 --label-smoothing 0.1 --warmup 0.1 \
--batch-size 256 --n-epochs 500 --optimizer-type adamax \
--patience 100 --beam-size 40 --add-segment-embed --add-inverse-rels \
--seed 12548 > ckpts/CKPT_DIR/log.txt 2>&1
```

## Evaluation (specify split)
```
python eval.py --dataset-path <DATA_PATH> --cuda \
--ckpt-path ckpts/CKPT_DIR/model.pt \
--split <valid/test> --sampling-type minerva \
--graph-connection type_1 --embed-dim 320 --n-attn-heads 8 \
--n-bert-layers 3
```


## Citation
```
@inproceedings{DBLP:journals/corr/abs-2212-09724,
  author       = {Vardaan Pahuja and
                  Boshi Wang and
                  Hugo Latapie and
                  Jayanth Srinivasa and
                  Yu Su},
  title        = {A Retrieve-and-Read Framework for Knowledge Graph Link Prediction},
  booktitle    = {Proceedings of the 32nd {ACM} International Conference on Information
                  {\&} Knowledge Management},
  journal      = {Conference on Information and Knowledge Managament (CIKM)},
  year         = {2023},
  url          = {https://arxiv.org/abs/2212.09724},
  doi          = {10.48550/arXiv.2212.09724},
  abbr = {CIKM},
  publisher    = {{ACM}},
  pdf={https://arxiv.org/abs/2212.09724}
}              
```